"""
Paso 8: scoring.py — xRV neutral y escala Stuff+.

Junta las probabilidades del modelo (predicciones_12conteos.parquet) con los
precios en carreras (rv_tabla.csv) y los pesos de la liga (pesos_conteo.csv):

    xRV(pitch, conteo) = sum_k P(k | pitch, conteo) * RV_k(conteo)
    xRV_Stuff(pitch)   = sum_conteo w_conteo * xRV(pitch, conteo)      (neutral)
    Stuff+             = 100 + 10 * z,  z = (-xRV_Stuff - mu) / sigma   (por temporada)

Convención: xRV positivo = bueno para el bateador; Stuff+ alto = mejor pitcher.

Esta v1 usa las predicciones out-of-fold del entrenamiento (2024+2025), así que
el Stuff+ es honesto (cada pitch lo predijo un modelo que no vio a su pitcher).
El `main` hace el contexto neutral (Stuff+); el contexto real (Location+) lo usa
scoring_location.py. El `in_play` usa el RV_k v0 (lo afinará
el modelo de contacto, paso 7).

Uso:
    python scoring.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from limpieza import cargar_datos
from objetivo import CLASES, agregar_resultado
from particion import separar_test

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_SALIDAS = RAIZ / "salidas"

COLS_PROB = [f"p_{c}" for c in CLASES]
COLS_RV = [f"rv_{c}" for c in CLASES]


def cargar_rv():
    """RV_k por conteo (ancho) y pesos de la liga, con columnas Balls/Strikes."""
    rv = pd.read_csv(CARPETA_SALIDAS / "rv_tabla.csv")
    rv_wide = rv.pivot_table(index=["balls", "strikes"], columns="outcome_class", values="rv")[CLASES]
    rv_wide.columns = COLS_RV
    rv_wide = rv_wide.reset_index().rename(columns={"balls": "Balls", "strikes": "Strikes"})
    pesos = (pd.read_csv(CARPETA_SALIDAS / "pesos_conteo.csv")
             .rename(columns={"balls": "Balls", "strikes": "Strikes"}))
    return rv_wide, pesos


def calcular_xrv(df12, contexto="neutral", promedio_conteo=None):
    """xRV por pitch.

    contexto='neutral' (Stuff+): df12 trae las predicciones en los 12 conteos; se
    promedian con los pesos de la liga. Devuelve PitchUID, pitcher_anon_id, xrv_stuff.

    contexto='real' (Location+/Pitching+): df12 trae UNA fila por pitch, con las
    probabilidades en su conteo real. xRV = sum_k P(k) * RV_k(conteo real), y se le
    resta el xRV promedio de ese conteo (lo que vale estar en ese conteo no es mérito
    del pitch). Devuelve PitchUID, pitcher_anon_id, Balls, Strikes, xrv, xrv_conteo
    (el promedio restado) y xrv_vs_conteo = xrv - xrv_conteo.
    `promedio_conteo`: tabla Balls, Strikes, xrv_conteo. Por defecto se calcula con df12
    (la liga); para pitches nuevos (dashboard) se pasa la de la liga.
    """
    if contexto == "real":
        return _xrv_real(df12, promedio_conteo)
    if contexto != "neutral":
        raise ValueError(f"contexto desconocido: {contexto!r} (usar 'neutral' o 'real')")
    rv_wide, pesos = cargar_rv()
    df = df12.copy()
    df["Balls"] = df["Balls"].astype(int)
    df["Strikes"] = df["Strikes"].astype(int)
    df = df.merge(rv_wide, on=["Balls", "Strikes"], how="left").merge(pesos, on=["Balls", "Strikes"], how="left")

    xrv_conteo = (df[COLS_PROB].to_numpy() * df[COLS_RV].to_numpy()).sum(axis=1)
    df["xrv_pond"] = xrv_conteo * df["peso"].to_numpy()          # aporte de cada conteo
    return (df.groupby("PitchUID", sort=False)
            .agg(pitcher_anon_id=("pitcher_anon_id", "first"), xrv_stuff=("xrv_pond", "sum"))
            .reset_index())


def _xrv_real(df, promedio_conteo=None):
    """Contexto 'real' de calcular_xrv: xRV en el conteo real, menos el promedio del conteo."""
    rv_wide, _pesos = cargar_rv()
    df = df[["PitchUID", "pitcher_anon_id", "Balls", "Strikes"] + COLS_PROB].copy()
    df["Balls"] = df["Balls"].astype(int)
    df["Strikes"] = df["Strikes"].astype(int)
    df = df.merge(rv_wide, on=["Balls", "Strikes"], how="left", validate="many_to_one")
    assert df[COLS_RV].notna().all().all(), "hay conteos sin precio RV_k"
    df["xrv"] = (df[COLS_PROB].to_numpy() * df[COLS_RV].to_numpy()).sum(axis=1)

    if promedio_conteo is None:
        promedio_conteo = df.groupby(["Balls", "Strikes"], as_index=False).agg(xrv_conteo=("xrv", "mean"))
    df = df.merge(promedio_conteo, on=["Balls", "Strikes"], how="left", validate="many_to_one")
    assert df["xrv_conteo"].notna().all(), "hay conteos sin promedio"
    df["xrv_vs_conteo"] = df["xrv"] - df["xrv_conteo"]
    return df[["PitchUID", "pitcher_anon_id", "Balls", "Strikes", "xrv", "xrv_conteo", "xrv_vs_conteo"]]


def a_escala_plus(s, year):
    """Stuff+ = 100 + 10z, con media y desviación POR TEMPORADA (+ = mejor pitcher)."""
    d = pd.DataFrame({"s": np.asarray(s, dtype=float), "year": np.asarray(year)})
    g = d.groupby("year")["s"]
    return (100 + 10 * (d["s"] - g.transform("mean")) / g.transform("std")).to_numpy()


MIN_PITCHES_ESCALA = 300    # pitchers con al menos esto definen la media y la sd de la escala por pitcher


def a_escala_pitcher(medias, n, min_n=MIN_PITCHES_ESCALA):
    """Score de PITCHER en escala 100 + 10z entre pitchers (100 = pitcher promedio, 110 = una sd mejor).

    Los scores por pitch (a_escala_plus) tienen sd 10 entre PITCHES; el promedio de un pitcher varía mucho menos
    (Location+ y Pitching+: sd ≈ 0.9 entre pitchers con muestra, porque solo ≈ 1 % de la varianza entre pitches es
    del pitcher: el resto es dónde cayó cada pitch). Para comparar pitchers se reescala con la media y la sd de los
    pitchers con >= `min_n` pitches (cada pitcher cuenta una vez). No cambia el orden de los pitchers.
    """
    medias = pd.Series(np.asarray(medias, dtype=float))
    elegibles = medias[np.asarray(n) >= min_n]
    return (100 + 10 * (medias - elegibles.mean()) / elegibles.std()).to_numpy()


def cargar_stuff_oficial():
    """Stuff+ oficial por pitch para las revisiones cruzadas: v3 a nivel del mar si ya se corrió, si no v1 (con su nombre)."""
    ruta = CARPETA_SALIDAS / "scores_stuff_altitud_pitch.parquet"
    if ruta.exists():
        return (pd.read_parquet(ruta, columns=["PitchUID", "stuff_plus_mar"]).rename(columns={"stuff_plus_mar": "stuff_plus"}),
                "Stuff+ v3")
    return pd.read_parquet(CARPETA_SALIDAS / "scores_pitch.parquet", columns=["PitchUID", "stuff_plus"]), "Stuff+ v1"


def main():
    datos = agregar_resultado(cargar_datos()).reset_index(drop=True)
    entrenamiento, _test = separar_test(datos)
    df12 = pd.read_parquet(CARPETA_SALIDAS / "predicciones_12conteos.parquet")

    xrv = calcular_xrv(df12, "neutral")                          # PitchUID, pitcher_anon_id, xrv_stuff
    meta = entrenamiento[["PitchUID", "AutoPitchType", "year", "RelSpeed", "SpinRate", "outcome_class"]]
    scores = xrv.merge(meta, on="PitchUID", how="left", validate="one_to_one")
    assert scores["year"].notna().all(), "hay pitches de las predicciones que no están en el entrenamiento"
    scores["stuff_plus"] = a_escala_plus(-scores["xrv_stuff"], scores["year"])
    scores["es_whiff"] = (scores["outcome_class"] == "whiff").astype(int)

    # ---- salidas (contrato seccion 6)
    CARPETA_SALIDAS.mkdir(exist_ok=True)
    (scores[["PitchUID", "pitcher_anon_id", "AutoPitchType", "year", "xrv_stuff", "stuff_plus"]]
     .to_parquet(CARPETA_SALIDAS / "scores_pitch.parquet", index=False))
    por_tipo = (scores.groupby(["pitcher_anon_id", "AutoPitchType"])
                .agg(stuff_plus=("stuff_plus", "mean"), n=("stuff_plus", "size")).reset_index())
    por_tipo.to_csv(CARPETA_SALIDAS / "scores_pitcher_tipo.csv", index=False)
    por_pitcher = (scores.groupby("pitcher_anon_id")                      # ponderado por uso = media de sus pitches
                   .agg(stuff_plus=("stuff_plus", "mean"), n=("stuff_plus", "size")).reset_index())
    por_pitcher["stuff_plus_pitcher"] = a_escala_pitcher(por_pitcher["stuff_plus"], por_pitcher["n"])   # 100 ± 10 entre pitchers
    por_pitcher.to_csv(CARPETA_SALIDAS / "scores_pitcher.csv", index=False)

    # ---- Sanity 1: por construccion, media ~100 y sd ~10 en cada temporada
    print("Sanity 1 - Stuff+ por temporada (media ~100, sd ~10 por construccion):")
    print(scores.groupby("year")["stuff_plus"].agg(["mean", "std", "count"]).round(2).to_string())

    # ---- Sanity 2: a nivel pitcher, mas Stuff+ deberia ir con mas whiffs
    pw = (scores.groupby("pitcher_anon_id")
          .agg(stuff=("stuff_plus", "mean"), whiff=("es_whiff", "mean"), n=("es_whiff", "size")))
    pw = pw[pw["n"] >= 200]
    print(f"\nSanity 2 - correlacion (nivel pitcher, n>=200: {len(pw)} pitchers) "
          f"Stuff+ vs tasa de whiff: r = {pw['stuff'].corr(pw['whiff']):.3f}")

    # ---- Sanity 3: leaderboard pitcher x tipo con fisica (debe tener sentido)
    fis = (scores.groupby(["pitcher_anon_id", "AutoPitchType"])
           .agg(stuff=("stuff_plus", "mean"), n=("stuff_plus", "size"),
                velo=("RelSpeed", "mean"), spin=("SpinRate", "mean"), whiff=("es_whiff", "mean")))
    fis = fis[fis["n"] >= 100].sort_values("stuff", ascending=False)
    fis["whiff_%"] = (100 * fis.pop("whiff")).round(1)
    fis[["stuff", "velo", "spin"]] = fis[["stuff", "velo", "spin"]].round(1)
    print("\nSanity 3 - top 8 pitcher x tipo por Stuff+ (n>=100):")
    print(fis.head(8).to_string())
    print("\n   bottom 8:")
    print(fis.tail(8).to_string())

    print(f"\nGuardado en {CARPETA_SALIDAS}: scores_pitch.parquet, scores_pitcher_tipo.csv, scores_pitcher.csv")


if __name__ == "__main__":
    main()
