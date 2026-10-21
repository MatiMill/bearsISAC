"""
Pitching+: de las probabilidades del modelo que ve todo al score 100 + 10z.

Igual que Location+ (scoring_location.py) pero con las predicciones de
pitching_plus.py (una fila por pitch, en su conteo REAL):

    xRV(pitch)    = sum_k P(k | todo, conteo real) * RV_k(conteo real)
    xRV_vs_conteo = xRV - xRV promedio de la liga en ese conteo
    Pitching+     = 100 + 10 * z,  z = (-xRV_vs_conteo - mu) / sigma   (por temporada)

Se resta el promedio del conteo por la misma razón que en Location+: lo que vale
estar en 0-2 vs 3-0 es del conteo, no del lanzamiento. Pitching+ mide la ejecución
TOTAL (stuff + ubicación + comando), así que debe correlacionar con Stuff+ y Location+
(chequeo de la guía §4).

Convención: xRV positivo = bueno para el bateador; Pitching+ alto = mejor pitcher.
v1: out-of-fold sobre el entrenamiento (2024+2025), `in_play` con RV_k v0.

Uso:
    python scoring_pitching.py      # antes: python pitching_plus.py
"""
import pandas as pd

from limpieza import cargar_datos
from objetivo import agregar_resultado
from particion import separar_test
from scoring import CARPETA_SALIDAS, a_escala_pitcher, a_escala_plus, calcular_xrv, cargar_stuff_oficial
from zona import preparar_ubicacion

MIN_PITCHES = 200       # pitchers con al menos esto para las revisiones a nivel pitcher


def estabilidad_entre_temporadas(scores, col):
    """Correlación del score medio de cada pitcher en 2024 vs 2025 (>= MIN_PITCHES cada año)."""
    por_año = scores.groupby(["pitcher_anon_id", "year"])[col].agg(["mean", "size"])
    por_año = por_año[por_año["size"] >= MIN_PITCHES]["mean"].unstack("year").dropna()
    años = sorted(por_año.columns)
    return por_año[años[0]].corr(por_año[años[1]]), len(por_año)


def main():
    datos = preparar_ubicacion(agregar_resultado(cargar_datos()))
    entrenamiento, _test = separar_test(datos)
    oof = pd.read_parquet(CARPETA_SALIDAS / "predicciones_oof_pitching.parquet")

    xrv = calcular_xrv(oof, "real")
    meta = entrenamiento[["PitchUID", "AutoPitchType", "year", "outcome_class"]]
    scores = xrv.merge(meta, on="PitchUID", how="left", validate="one_to_one")
    assert scores["year"].notna().all(), "hay pitches del OOF que no están en el entrenamiento"
    scores["pitching_plus"] = a_escala_plus(-scores["xrv_vs_conteo"], scores["year"])

    # ---- salidas
    CARPETA_SALIDAS.mkdir(exist_ok=True)
    (scores[["PitchUID", "pitcher_anon_id", "AutoPitchType", "year", "Balls", "Strikes",
             "xrv", "xrv_vs_conteo", "pitching_plus"]]
     .rename(columns={"xrv": "xrv_pitching"})
     .to_parquet(CARPETA_SALIDAS / "scores_pitching_pitch.parquet", index=False))
    (scores.groupby(["Balls", "Strikes"], as_index=False)["xrv_conteo"].first()
     .to_csv(CARPETA_SALIDAS / "xrv_promedio_conteo_pitching.csv", index=False))
    por_tipo = (scores.groupby(["pitcher_anon_id", "AutoPitchType"])
                .agg(pitching_plus=("pitching_plus", "mean"), n=("pitching_plus", "size")).reset_index())
    por_tipo.to_csv(CARPETA_SALIDAS / "scores_pitching_pitcher_tipo.csv", index=False)
    por_pitcher = (scores.groupby("pitcher_anon_id")                  # ponderado por uso = media de sus pitches
                   .agg(pitching_plus=("pitching_plus", "mean"), n=("pitching_plus", "size")).reset_index())
    por_pitcher["pitching_plus_pitcher"] = a_escala_pitcher(por_pitcher["pitching_plus"], por_pitcher["n"])   # 100 ± 10 entre pitchers
    por_pitcher.to_csv(CARPETA_SALIDAS / "scores_pitching_pitcher.csv", index=False)

    # ---- Sanity 1: media ~100 / sd ~10 por temporada
    print("Sanity 1 - Pitching+ por temporada (media ~100, sd ~10 por construcción):")
    print(scores.groupby("year")["pitching_plus"].agg(["mean", "std", "count"]).round(2).to_string())

    # ---- Sanity 2: restar el promedio deja cada conteo centrado en 0
    centro = scores.groupby(["Balls", "Strikes"])["xrv_vs_conteo"].mean().abs().max()
    print(f"\nSanity 2 - máx |media de xRV_vs_conteo| entre los 12 conteos: {centro:.1e} (debe ser ~0)")

    # ---- Sanity 3: Pitching+ debe ir con MENOS bolas y con MÁS whiffs (combina ubicación y stuff)
    scores["es_ball"] = (scores["outcome_class"] == "ball").astype(int)
    scores["es_whiff"] = (scores["outcome_class"] == "whiff").astype(int)
    pp = (scores.groupby("pitcher_anon_id")
          .agg(pit=("pitching_plus", "mean"), ball=("es_ball", "mean"),
               whiff=("es_whiff", "mean"), n=("pitching_plus", "size")))
    pp = pp[pp["n"] >= MIN_PITCHES]
    print(f"\nSanity 3 - a nivel pitcher (n>={MIN_PITCHES}, {len(pp)} pitchers): "
          f"Pitching+ vs tasa de bolas r = {pp['pit'].corr(pp['ball']):.3f} (neg); "
          f"vs tasa de whiff r = {pp['pit'].corr(pp['whiff']):.3f} (pos)")

    # ---- Sanity 4 (guía §4): debe correlacionar con Stuff+ y con Location+
    stuff, nombre_stuff = cargar_stuff_oficial()
    loc = pd.read_parquet(CARPETA_SALIDAS / "scores_location_pitch.parquet")[["PitchUID", "location_plus"]]
    tri = (scores[["PitchUID", "pitcher_anon_id", "pitching_plus"]]
           .merge(stuff, on="PitchUID", how="inner").merge(loc, on="PitchUID", how="inner"))
    g = (tri.groupby("pitcher_anon_id")
         .agg(pit=("pitching_plus", "mean"), stuff=("stuff_plus", "mean"),
              loc=("location_plus", "mean"), n=("pitching_plus", "size")))
    g = g[g["n"] >= MIN_PITCHES]
    print(f"\nSanity 4 - Pitching+ vs {nombre_stuff} r = {g['pit'].corr(g['stuff']):.3f}; "
          f"vs Location+ r = {g['pit'].corr(g['loc']):.3f} "
          f"(debe correlacionar con los dos; {len(g)} pitchers)")
    print(f"   referencia: {nombre_stuff} vs Location+ r = {g['stuff'].corr(g['loc']):.3f}")

    # ---- Sanity 5: estabilidad entre temporadas
    r, n = estabilidad_entre_temporadas(scores, "pitching_plus")
    print(f"\nSanity 5 - estabilidad Pitching+ 2024 -> 2025 ({n} pitchers con >= {MIN_PITCHES} en cada año): r = {r:.3f}")

    print(f"\nGuardado en {CARPETA_SALIDAS}: scores_pitching_pitch.parquet, scores_pitching_pitcher_tipo.csv, "
          f"scores_pitching_pitcher.csv, xrv_promedio_conteo_pitching.csv")


if __name__ == "__main__":
    main()
