"""
A quién contratar (en teoría): ranking de pitchers para Harp Helú.

Idea: un pitcher vale por (1) lo bien que ejecuta —Pitching+, que ve física + ubicación— y (2) cuánto de eso conserva en
el aire de Harp Helú, según el contrafactual de altitud (contrafactual_altitud.py). Todo en las mismas unidades,
CARRERAS POR 100 PITCHES contra un pitcher promedio (+ = mejor):

    rating = ejecución_ajustada + ajuste_de_altitud

- ejecución = −xRV_vs_conteo medio de Pitching+ (scores_pitching_pitch.parquet), temporadas 2024+2025 juntas.
  Con pocos pitches el promedio es ruidoso: se ENCOGE hacia el promedio de la liga con la confiabilidad
  R = var_real / (var_real + var_ruido / n), estimada de los propios datos (componentes de varianza).
- ajuste de altitud = delta_relativo del contrafactual (puntos de Stuff+ que ese pitcher pierde de MÁS o de MENOS que un pitch
  promedio al pasar a Harp Helú: su mezcla de pitches y su movimiento) × carreras por punto (σ de Stuff+ / 10). Es un
  cálculo del modelo: NO está validado contra resultados reales (no hay muestra suficiente de los mismos pitchers
  en Harp Helú y fuera).
- Pitchers de los Diablos (los que lanzan de local en Harp Helú) se marcan para poder excluirlos.

Por qué Pitching+ como base: se probó cuál score de 2024 predice mejor el resultado de 2025 (K−BB% y carreras por turno,
ajustadas por parque); Pitching+ fue el mejor predictor de ambos y Stuff+ solo casi no predijo K−BB%. Ver `prueba_retrospectiva`.

La confiabilidad R solo descuenta el ruido de la muestra; no incluye que el talento cambie de un año a otro (la
estabilidad real de Pitching+ entre 2024 y 2025 es r = 0.60), así que la incertidumbre es un MÍNIMO.

Honestidad: la prueba retrospectiva mide cuánto aciertan los rankings de un año al siguiente; no es un experimento de
contratación (no hay datos de pitchers que cambian de equipo). No considera salud, edad, contrato, rol ni disponibilidad.

Salidas (fuera del repo):
    ../salidas/ranking_contratacion.csv   una fila por pitcher
    ../salidas/prueba_retrospectiva.json  números de la prueba 2024 → 2025

Uso:
    python contratacion.py      # antes: scoring_pitching.py, contrafactual_altitud.py (y sus requisitos)
"""
import json

import numpy as np
import pandas as pd

from limpieza import RUTA
from modelo_multiclase import CARPETA_SALIDAS
from run_values import COLUMNAS, armar_turnos, carreras_esperadas, valor_de_turnos

MIN_PITCHES = 300           # pitches mínimos (2024+2025) para entrar al ranking
MIN_PITCHES_REL = 100       # pitchers que se usan para estimar la varianza real y la del ruido
MIN_TURNOS = 50             # turnos mínimos en la prueba retrospectiva
RUTA_RANKING = CARPETA_SALIDAS / "ranking_contratacion.csv"
RUTA_PRUEBA = CARPETA_SALIDAS / "prueba_retrospectiva.json"


# ------------------------------------------------------------------ resultados reales por turno
def turnos_por_pitcher():
    """Una fila por pitcher × temporada con turnos, K−BB% y carreras por turno (ajustadas por parque; + = mejor).

    Las carreras por turno son el valor de run_values.py (RE después − RE antes + carreras) menos el promedio de ese parque
    y temporada, para no castigar a los pitchers que lanzan en Harp Helú. Solo temporadas de entrenamiento.
    """
    df = pd.read_parquet(RUTA, columns=COLUMNAS + ["pitcher_anon_id"])
    df = df[(df["year"] != df["year"].max()) & (df["Outs"] <= 2)].reset_index(drop=True)
    for c in ["Outs", "Balls", "Strikes", "RunsScored"]:
        df[c] = df[c].astype(int)
    df["media_entrada"] = df.groupby(["game_anon_id", "Inning", "Top/Bottom"], observed=True).ngroup()
    df["es_final"] = df["KorBB"].isin(["Strikeout", "Walk"]) | df["PitchCall"].isin(["InPlay", "HitByPitch"])
    turnos = armar_turnos(df)
    turnos = valor_de_turnos(df, turnos, carreras_esperadas(turnos))
    meta = df.groupby("turno").agg(game=("game_anon_id", "first"), pitcher=("pitcher_anon_id", "first"), year=("year", "first"))
    t = turnos.join(meta)
    t = t[t["terminado"] & t["pitcher"].notna() & t["valor"].notna()].copy()
    est = pd.read_parquet(RUTA.parent / "estadios.parquet")[["game_anon_id", "estadio", "altitud"]].rename(columns={"game_anon_id": "game"})
    t = t.merge(est, on="game", how="left")
    t["parque"] = t["estadio"].fillna(t["altitud"].astype(str))
    t["valor_adj"] = t["valor"] - t.groupby(["parque", "year"])["valor"].transform("mean")
    t["K"], t["BB"] = (t["evento"] == "Strikeout").astype(int), (t["evento"] == "Walk").astype(int)
    g = t.groupby(["pitcher", "year"]).agg(turnos=("valor", "size"), K=("K", "mean"), BB=("BB", "mean"), runs_pa=("valor_adj", "mean")).reset_index()
    g["kbb"] = g["K"] - g["BB"]
    g["runs_pa"] = -g["runs_pa"]
    return g.rename(columns={"pitcher": "pitcher_anon_id"})


# ------------------------------------------------------------------ encogimiento por poca muestra
def componentes_de_varianza(x, grupo):
    """(media de la liga, varianza real entre pitchers, varianza de ruido por pitch) del valor por pitch x."""
    g = x.groupby(grupo)
    n, m, v = g.size(), g.mean(), g.var(ddof=1)
    ok = n >= MIN_PITCHES_REL
    var_ruido = float(((n[ok] - 1) * v[ok]).sum() / (n[ok] - 1).sum())
    media = float(x.mean())
    var_real = float(max(np.average((m[ok] - media) ** 2, weights=n[ok]) - np.average(var_ruido / n[ok], weights=n[ok]), 1e-12))
    return media, var_real, var_ruido


def encoger(m, n, media, var_real, var_ruido):
    """Promedio encogido hacia la media de la liga y la confiabilidad R de cada pitcher."""
    r = var_real / (var_real + var_ruido / n)
    return media + r * (m - media), r


# ------------------------------------------------------------------ ranking
def construir_ranking():
    pit = pd.read_parquet(CARPETA_SALIDAS / "scores_pitching_pitch.parquet")
    pit["valor"] = -pit["xrv_vs_conteo"]                                    # + = bueno para el pitcher, carreras por pitch
    alt = pd.read_parquet(CARPETA_SALIDAS / "scores_stuff_altitud_pitch.parquet")
    loc = pd.read_parquet(CARPETA_SALIDAS / "scores_location_pitch.parquet")[["PitchUID", "location_plus"]]
    calib = json.loads((CARPETA_SALIDAS / "calibracion_stuff_altitud.json").read_text(encoding="utf-8"))
    runs_por_punto = calib["sd"] / 10.0                                      # carreras por pitch que vale 1 punto de Stuff+

    media, var_real, var_ruido = componentes_de_varianza(pit["valor"], pit["pitcher_anon_id"])
    g = pit.groupby("pitcher_anon_id").agg(n_pitches=("valor", "size"), ejecucion=("valor", "mean"), pitching_plus=("pitching_plus", "mean"))
    g["ejecucion_ajustada"], g["confianza"] = encoger(g["ejecucion"], g["n_pitches"], media, var_real, var_ruido)
    # incertidumbre (1 desviación estándar, carreras por 100 pitches) de la ejecución real: var_real · (1 − R)
    g["incertidumbre_100"] = 100 * np.sqrt(var_real * (1 - g["confianza"]))

    a = alt.groupby("pitcher_anon_id").agg(stuff_mar=("stuff_plus_mar", "mean"), stuff_harp=("stuff_plus_harp", "mean"),
                                           delta_relativo=("delta_relativo", "mean"))
    loc = pit[["PitchUID", "pitcher_anon_id"]].merge(loc, on="PitchUID").groupby("pitcher_anon_id")["location_plus"].mean()
    mix = (alt.groupby(["pitcher_anon_id", "AutoPitchType"], observed=True).size().rename("n").reset_index())
    mix["pct"] = 100 * mix["n"] / mix.groupby("pitcher_anon_id")["n"].transform("sum")
    mix = mix.sort_values(["pitcher_anon_id", "pct"], ascending=[True, False])
    texto = mix[mix["pct"] >= 8].groupby("pitcher_anon_id").apply(
        lambda d: ", ".join(f"{t} {p:.0f}%" for t, p in zip(d["AutoPitchType"], d["pct"])), include_groups=False)
    g = g.join(a).join(loc).join(texto.rename("arsenal"))
    g["tipo_principal"] = mix.drop_duplicates("pitcher_anon_id").set_index("pitcher_anon_id")["AutoPitchType"]

    g["ajuste_altitud"] = g["delta_relativo"] * runs_por_punto
    g["rating_100"] = 100 * (g["ejecucion_ajustada"] + g["ajuste_altitud"])            # carreras por 100 pitches vs promedio
    g["ejecucion_100"] = 100 * g["ejecucion_ajustada"]
    g["ajuste_altitud_100"] = 100 * g["ajuste_altitud"]

    # Resultados reales (lo que el coach ya conoce) y marca de los Diablos
    r = turnos_por_pitcher()
    tot = r.groupby("pitcher_anon_id").apply(
        lambda d: pd.Series({"turnos": d["turnos"].sum(), "kbb": np.average(d["kbb"], weights=d["turnos"]),
                             "runs_pa": np.average(d["runs_pa"], weights=d["turnos"])}), include_groups=False)
    g = g.join(tot)
    g["diablos"] = g.index.isin(pitchers_diablos())

    g = g[g["n_pitches"] >= MIN_PITCHES].copy()
    z = (g["rating_100"] - g["rating_100"].mean()) / g["rating_100"].std()
    g["indice"] = 100 + 10 * z
    g = g.sort_values("rating_100", ascending=False)
    g["puesto"] = np.arange(1, len(g) + 1)
    return g.reset_index().rename(columns={"index": "pitcher_anon_id"}), {"media": media, "var_real": var_real, "var_ruido": var_ruido,
                                                                           "runs_por_punto": runs_por_punto}


def pitchers_diablos():
    """Pitchers que lanzan DE LOCAL en juegos de Harp Helú (lanza el local en la parte alta): el plantel de los Diablos."""
    from efecto_altitud import agregar_densidad
    df = pd.read_parquet(RUTA, columns=["game_anon_id", "pitcher_anon_id", "Top/Bottom", "year"])
    df = df[df["year"] != df["year"].max()]
    df = agregar_densidad(df)
    hh = df[df["harp_helu"]]
    local = hh["Top/Bottom"].astype(str).str.lower().str.startswith("top")
    cuenta = pd.DataFrame({"local": hh[local].groupby("pitcher_anon_id").size(), "visita": hh[~local].groupby("pitcher_anon_id").size()}).fillna(0)
    return set(cuenta.index[cuenta["local"] / (cuenta["local"] + cuenta["visita"]) >= 0.5])


# ------------------------------------------------------------------ prueba retrospectiva
def prueba_retrospectiva():
    """¿El ranking que se hace con 2024 acierta en 2025? Compara contra lo que de verdad pasó (K−BB% y carreras por turno)."""
    pit = pd.read_parquet(CARPETA_SALIDAS / "scores_pitching_pitch.parquet")
    pit["valor"] = -pit["xrv_vs_conteo"]
    p24 = pit[pit["year"] == 2024]
    media, var_real, var_ruido = componentes_de_varianza(p24["valor"], p24["pitcher_anon_id"])
    g = p24.groupby("pitcher_anon_id").agg(n=("valor", "size"), m=("valor", "mean"))
    g["proy"], _ = encoger(g["m"], g["n"], media, var_real, var_ruido)
    r = turnos_por_pitcher()
    r24, r25 = r[r["year"] == 2024].set_index("pitcher_anon_id"), r[r["year"] == 2025].set_index("pitcher_anon_id")
    w = g[g["n"] >= 200].join(r24[["turnos", "kbb", "runs_pa"]].add_suffix("_24")).join(r25[["turnos", "kbb", "runs_pa"]].add_suffix("_25"), how="inner")
    w = w[(w["turnos_24"] >= MIN_TURNOS) & (w["turnos_25"] >= MIN_TURNOS)].dropna()
    salida = {"n_pitchers": int(len(w))}
    for meta, nombre in [("kbb", "K−BB%"), ("runs_pa", "carreras por turno")]:
        y = w[f"{meta}_25"]
        r_rank, r_res = float(w["proy"].corr(y)), float(w[f"{meta}_24"].corr(y))
        top = w["proy"].nlargest(20).index
        n = len(w)
        z = np.arctanh(r_rank)
        salida[meta] = {"nombre": nombre, "r_ranking": r_rank, "r_ranking_ic95": [float(np.tanh(z - 1.96 / np.sqrt(n - 3))), float(np.tanh(z + 1.96 / np.sqrt(n - 3)))],
                        "r_resultado_pasado": r_res,
                        "top20_2025": float(y.loc[top].mean()), "resto_2025": float(y.drop(top).mean()), "todos_2025": float(y.mean()),
                        "peor20_2025": float(y.loc[w["proy"].nsmallest(20).index].mean())}
    return salida


def main():
    ranking, comp = construir_ranking()
    CARPETA_SALIDAS.mkdir(exist_ok=True)
    ranking.to_csv(RUTA_RANKING, index=False)
    print(f"Varianza real entre pitchers: {comp['var_real']:.3e} | ruido por pitch: {comp['var_ruido']:.3e} "
          f"→ con 1,000 pitches la confiabilidad es {comp['var_real'] / (comp['var_real'] + comp['var_ruido'] / 1000):.2f}")
    print(f"Pitchers en el ranking (≥ {MIN_PITCHES} pitches): {len(ranking)} | Diablos: {int(ranking['diablos'].sum())}")
    print(f"Ejecución (carreras/100 pitches): sd {ranking['ejecucion_100'].std():.3f} | ajuste de altitud: sd {ranking['ajuste_altitud_100'].std():.3f}")

    cols = ["puesto", "pitcher_anon_id", "rating_100", "ejecucion_100", "ajuste_altitud_100", "confianza", "n_pitches", "stuff_harp", "location_plus", "pitching_plus", "arsenal"]
    print("\nTop 10 (sin Diablos):")
    print(ranking[~ranking["diablos"]].head(10)[cols].round(3).to_string(index=False))
    print("\nTop 5 de los Diablos (referencia):")
    print(ranking[ranking["diablos"]].head(5)[cols].round(3).to_string(index=False))

    prueba = prueba_retrospectiva()
    RUTA_PRUEBA.write_text(json.dumps(prueba, indent=2), encoding="utf-8")
    print(f"\nPrueba retrospectiva (ranking con 2024 → resultado de 2025; {prueba['n_pitchers']} pitchers):")
    for meta in ["kbb", "runs_pa"]:
        d = prueba[meta]
        esc = 100 if meta == "runs_pa" else 100
        print(f"  {d['nombre']:20s} r ranking→2025 = {d['r_ranking']:.2f} (IC95 {d['r_ranking_ic95'][0]:.2f} a {d['r_ranking_ic95'][1]:.2f}) | "
              f"resultado pasado → 2025: r = {d['r_resultado_pasado']:.2f} | top 20: {esc * d['top20_2025']:+.2f} vs todos {esc * d['todos_2025']:+.2f} vs peor 20 {esc * d['peor20_2025']:+.2f} (por 100 turnos)")
    print(f"\nGuardado: {RUTA_RANKING} y {RUTA_PRUEBA}")


if __name__ == "__main__":
    main()
