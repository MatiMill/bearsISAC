"""
run_values.py  (paso 5 de docs/guia_proyecto.md)
Calcula el precio en carreras de cada resultado de un lanzamiento según el conteo:
RV_k(balls, strikes) para k en ball, called_strike, whiff, foul, in_play.

Convención de signo: positivo = bueno para el bateador.

Uso:
    python run_values.py

Guarda en ../salidas/ (fuera del repo, son datos del ISAC):
    rv_tabla.csv         balls, strikes, outcome_class, rv   (12 conteos x 5 resultados)
    valores_evento.csv   evento, n, valor                    (linear weights)
    pesos_conteo.csv     balls, strikes, peso                (fracción de pitches por conteo)

Reglas:
- Datos SIN limpiar: la limpieza quita filas que sí tienen carreras y outs reales.
- Solo temporadas de entrenamiento: la última temporada es el test sellado.
- Los datos no dicen qué corredores hay en base, pero las filas están en orden
  cronológico, así que se puede contar CUÁNTOS hay:
      corredores = bateadores que ya terminaron su turno - outs - carreras
  El estado de la entrada es (outs, número de corredores): 12 estados.
"""
import numpy as np
import pandas as pd

from limpieza import RUTA
from objetivo import CLASES
from particion import MODO_FINAL

CARPETA_SALIDAS = RUTA.parent / "salidas"
COLUMNAS = ["game_anon_id", "year", "Inning", "Top/Bottom", "batter_anon_id",
            "Outs", "Balls", "Strikes", "PitchCall", "KorBB", "play_result",
            "RunsScored", "OutsOnPlay", "hit_type"]


# ---------- PASO 0: datos crudos de entrenamiento, en orden ----------
def cargar_crudos():
    df = pd.read_parquet(RUTA, columns=COLUMNAS)   # el orden de las filas es cronológico
    temporada_test = df["year"].max()
    if not MODO_FINAL:                      # en modo final la temporada sellada ya se validó y entra
        df = df[df["year"] != temporada_test]
    df = df[df["Outs"] <= 2].reset_index(drop=True)
    for col in ["Outs", "Balls", "Strikes", "RunsScored"]:
        df[col] = df[col].astype(int)
    # Media entrada = un juego, una entrada, alta o baja
    df["media_entrada"] = df.groupby(["game_anon_id", "Inning", "Top/Bottom"], observed=True).ngroup()
    # Lanzamiento que termina el turno del bateador
    df["es_final"] = df["KorBB"].isin(["Strikeout", "Walk"]) | df["PitchCall"].isin(["InPlay", "HitByPitch"])
    return df, temporada_test


# ---------- PASO 1: turnos y estado de la entrada ----------
def armar_turnos(df):
    """Una fila por turno, con el estado (outs, corredores) al empezar y las carreras que entraron."""
    por_entrada = df.groupby("media_entrada", sort=False)
    # Empieza un turno nuevo si el pitch anterior terminó un turno o cambió el bateador
    anterior_final = por_entrada["es_final"].shift(fill_value=True)
    cambio_bateador = df["batter_anon_id"].ne(por_entrada["batter_anon_id"].shift())
    inicio = (anterior_final | cambio_bateador).astype(int)
    df["turno"] = df.groupby(["media_entrada", inicio.groupby(df["media_entrada"]).cumsum()]).ngroup()
    df["fila"] = df.index

    turnos = df.groupby("turno", sort=False).agg(
        media_entrada=("media_entrada", "first"),
        orden=("fila", "first"),                      # fila del primer pitch: da el orden
        outs=("Outs", "first"),
        carreras=("RunsScored", "sum"),               # todas las carreras durante el turno
        terminado=("es_final", "any"),
    ).sort_values("orden")

    por_entrada = turnos.groupby("media_entrada", sort=False)
    terminados_antes = por_entrada["terminado"].cumsum() - turnos["terminado"]
    carreras_antes = por_entrada["carreras"].cumsum() - turnos["carreras"]
    turnos["corredores"] = (terminados_antes - turnos["outs"] - carreras_antes).clip(0, 3)
    # Carreras desde que empieza este turno hasta que termina la entrada
    turnos["carreras_resto"] = por_entrada["carreras"].transform(lambda s: s[::-1].cumsum()[::-1])
    return turnos


# ---------- PASO 2: RE(outs, corredores) ----------
def carreras_esperadas(turnos):
    """RE(o, r): carreras promedio desde el estado (o outs, r corredores) hasta que termina la entrada."""
    return turnos.groupby(["outs", "corredores"])["carreras_resto"].mean()


# ---------- PASO 3: valor de cada turno ----------
def valor_de_turnos(df, turnos, re):
    """valor = RE(estado al terminar) - RE(estado al empezar) + carreras que entraron.

    El estado al terminar es el estado con el que empieza el siguiente turno de la
    misma entrada; si era el último turno, la entrada terminó y RE = 0.
    """
    estado = list(zip(turnos["outs"], turnos["corredores"]))
    turnos["re_inicio"] = [re.get(e, np.nan) for e in estado]
    turnos["re_fin"] = turnos.groupby("media_entrada", sort=False)["re_inicio"].shift(-1).fillna(0.0)
    turnos["valor"] = turnos["re_fin"] - turnos["re_inicio"] + turnos["carreras"]

    # Tipo de evento y conteo del pitch que terminó cada turno
    finales = df[df["es_final"]].drop_duplicates("turno", keep="last").set_index("turno")
    # HitByPitch primero: el PitchCall manda (22 pitches traen KorBB = Walk con PitchCall = HitByPitch)
    turnos["evento"] = pd.Series(np.select(
        [finales["PitchCall"] == "HitByPitch", finales["KorBB"] == "Strikeout", finales["KorBB"] == "Walk"],
        ["HitByPitch", "Strikeout", "Walk"],
        default=finales["play_result"].astype(str)), index=finales.index)
    turnos["pitch_final"] = finales["PitchCall"].astype(str)
    turnos["es_bunt"] = (finales["hit_type"].astype(str) == "Bunt").reindex(turnos.index, fill_value=False).astype(bool)
    turnos["balls_final"] = finales["Balls"]
    turnos["strikes_final"] = finales["Strikes"]
    return turnos


# ---------- PASO 4: V(conteo) ----------
def valor_por_conteo(df, turnos):
    """V(b, s): valor promedio de los turnos (terminados) que pasaron por el conteo b-s."""
    validos = turnos[turnos["terminado"]]
    # Pares únicos (turno, conteo): con 2 strikes los fouls repiten el mismo conteo
    pares = df.loc[df["turno"].isin(validos.index), ["turno", "Balls", "Strikes"]].drop_duplicates()
    pares["valor"] = pares["turno"].map(validos["valor"])
    v = pares.groupby(["Balls", "Strikes"])["valor"].mean()
    return {(b, s): v.loc[(b, s)] for b, s in v.index}


# ---------- PASO 5: RV_k(b, s) ----------
MIN_TURNOS_CONTEO = 50     # turnos mínimos para usar el valor específico de un conteo (si no, el promedio de la liga)


def _valor_por_conteo_final(turnos, evento, columna):
    """Valor promedio del `evento` según `columna` (balls_final o strikes_final); promedio de la liga si hay pocos turnos."""
    t = turnos[turnos["evento"] == evento]
    g = t.groupby(columna)["valor"].agg(["mean", "size"])
    return {k: (r["mean"] if r["size"] >= MIN_TURNOS_CONTEO else t["valor"].mean()) for k, r in g.iterrows()}


def tabla_rv(v, turnos, valores_evento, df):
    """RV_k(b, s). Tres detalles (auditoría 2026-10-21):

    - `in_play` NO incluye los bunts: la clase in_play del modelo los excluye (limpieza.py), así que su precio
      tampoco debe mezclarlos (los bunts siguen en V y RE: son turnos reales).
    - La 4.ª bola se paga con el valor de las bases por bolas que terminaron en ESE conteo (3-s) y el 3.er strike
      con el de los ponches que terminaron en b-2, no con el promedio de toda la liga (cae a ese promedio si hay
      < MIN_TURNOS_CONTEO turnos).
    - La clase `ball` incluye los pelotazos (objetivo.py): su precio mezcla el de una bola con el de un pelotazo,
      ponderado por la fracción de pelotazos entre los pitches de clase ball en ese conteo.
    """
    v_bb = _valor_por_conteo_final(turnos, "Walk", "strikes_final")       # base por bolas, por strikes (siempre 3 bolas)
    v_k = _valor_por_conteo_final(turnos, "Strikeout", "balls_final")     # ponche, por bolas (siempre 2 strikes)
    v_hbp = valores_evento.loc["HitByPitch", "valor"]
    sin_bunt = turnos[(turnos["pitch_final"] == "InPlay") & ~turnos["es_bunt"]]
    en_juego = sin_bunt.groupby(["balls_final", "strikes_final"])["valor"].mean()

    es_bola = df["PitchCall"].isin(["BallCalled", "BallinDirt", "HitByPitch"])
    frac_hbp = (df[es_bola].assign(hbp=lambda d: d["PitchCall"] == "HitByPitch")
                .groupby(["Balls", "Strikes"])["hbp"].mean())

    filas = []
    for b in range(4):
        for s in range(3):
            actual = v[(b, s)]
            bola = (v_bb[s] if b == 3 else v[(b + 1, s)]) - actual   # 4a bola = base por bolas
            strike = (v_k[b] if s == 2 else v[(b, s + 1)]) - actual  # 3er strike = ponche
            q = float(frac_hbp.get((b, s), 0.0))
            precios = {
                "ball": (1 - q) * bola + q * (v_hbp - actual),
                "called_strike": strike,
                "whiff": strike,
                "foul": 0.0 if s == 2 else strike,                    # foul con 2 strikes no cambia nada
                "in_play": en_juego.loc[(b, s)] - actual,
            }
            for clase in CLASES:
                filas.append({"balls": b, "strikes": s, "outcome_class": clase, "rv": precios[clase]})
    return pd.DataFrame(filas)


# ---------- PASO 6: pesos de cada conteo ----------
def pesos_conteo(df):
    pesos = df.groupby(["Balls", "Strikes"]).size() / len(df)
    return pesos.rename("peso").rename_axis(["balls", "strikes"]).reset_index()


def main():
    df, temporada_test = cargar_crudos()
    print(f"Temporadas de entrenamiento: {sorted(int(y) for y in df['year'].unique())} "
          f"(test sellado: {temporada_test})")
    print(f"Lanzamientos: {len(df):,} | medias entradas: {df['media_entrada'].nunique():,}\n")

    turnos = armar_turnos(df)
    print(f"Paso 1. Turnos: {len(turnos):,} ({turnos['terminado'].mean():.1%} terminados con un evento)")
    print("   Corredores en base al empezar el turno:",
          (turnos["corredores"].value_counts(normalize=True).sort_index() * 100).round(1).to_dict())

    re = carreras_esperadas(turnos)
    print("\nPaso 2. Carreras esperadas RE(outs, corredores):")
    print(re.unstack().round(3).to_string())

    turnos = valor_de_turnos(df, turnos, re)
    valores_evento = (turnos[turnos["terminado"]].groupby("evento")["valor"]
                      .agg(n="size", valor="mean").sort_values("valor"))
    print("\nPaso 3. Valor promedio de cada evento (linear weights):")
    print(valores_evento.round(3).to_string())

    v = valor_por_conteo(df, turnos)
    print("\nPaso 4. Valor por conteo V(b-s):")
    tabla_v = pd.DataFrame([{"balls": b, "strikes": s, "V": val} for (b, s), val in v.items()])
    print(tabla_v.pivot(index="balls", columns="strikes", values="V").round(3).to_string())

    rv = tabla_rv(v, turnos, valores_evento, df)
    print("\nPaso 5. Precio de cada resultado RV_k (+ = bueno para el bateador):")
    print(rv.pivot_table(index=["balls", "strikes"], columns="outcome_class", values="rv")[CLASES].round(3).to_string())

    pesos = pesos_conteo(df)

    print("\nPaso 6. Revisiones:")
    re0 = {o: re.loc[o].mean() for o in [0, 1, 2]}
    revisiones = {
        "RE baja con los outs": re0[0] > re0[1] > re0[2],
        "RE sube con los corredores": all(re.loc[o].is_monotonic_increasing for o in [0, 1, 2]),
        "V sube con las bolas": all(v[(b + 1, s)] > v[(b, s)] for b in range(3) for s in range(3)),
        "V baja con los strikes": all(v[(b, s + 1)] < v[(b, s)] for b in range(4) for s in range(2)),
        "Toda bola vale > 0": (rv.loc[rv["outcome_class"] == "ball", "rv"] > 0).all(),
        "Todo strike vale < 0": (rv.loc[rv["outcome_class"].isin(["called_strike", "whiff"]), "rv"] < 0).all(),
        "Los pesos suman 1": abs(pesos["peso"].sum() - 1) < 1e-9,
    }
    for nombre, ok in revisiones.items():
        print(f"   {'OK   ' if ok else 'FALLA'} {nombre}")

    CARPETA_SALIDAS.mkdir(exist_ok=True)
    rv.to_csv(CARPETA_SALIDAS / "rv_tabla.csv", index=False)
    valores_evento.reset_index().to_csv(CARPETA_SALIDAS / "valores_evento.csv", index=False)
    pesos.to_csv(CARPETA_SALIDAS / "pesos_conteo.csv", index=False)
    print(f"\nGuardado en {CARPETA_SALIDAS}: rv_tabla.csv, valores_evento.csv, pesos_conteo.csv")


if __name__ == "__main__":
    main()
