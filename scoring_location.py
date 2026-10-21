"""
Location+: de las probabilidades del modelo de ubicación al score 100 + 10z.

Junta las predicciones out-of-fold de modelo_location.py (una fila por pitch, en su
conteo REAL) con los precios en carreras (rv_tabla.csv):

    xRV(pitch)         = sum_k P(k | ubicación, conteo real, ...) * RV_k(conteo real)
    xRV_vs_conteo      = xRV - xRV promedio de la liga en ese conteo
    Location+          = 100 + 10 * z,  z = (-xRV_vs_conteo - mu) / sigma   (por temporada)

Por qué se resta el promedio del conteo: un pitch en 0-2 ya parte con ventaja para el
pitcher y uno en 3-0 con desventaja; eso es del conteo, no de dónde puso la bola. Restar
el promedio deja solo lo que aportó la ubicación DADO el conteo (en 3-0 conviene la zona,
en 0-2 conviene que persiga afuera).

Convención: xRV positivo = bueno para el bateador; Location+ alto = mejor pitcher.
v1: out-of-fold sobre el entrenamiento (2024+2025), `in_play` con RV_k v0.

Uso:
    python scoring_location.py      # antes: python modelo_location.py
"""
import pandas as pd

from limpieza import cargar_datos
from objetivo import agregar_resultado
from particion import separar_test
from scoring import CARPETA_SALIDAS, a_escala_pitcher, a_escala_plus, calcular_xrv, cargar_stuff_oficial
from zona import NOMBRES_ZONA, preparar_ubicacion

MIN_PITCHES = 200       # pitchers con al menos esto para las revisiones a nivel pitcher


def revisar_zona_por_conteo(scores):
    """Location+ promedio por zona en 3-0 (hay que meterla) y en 0-2 (conviene que persiga)."""
    sub = scores[scores["conteo"].isin(["3-0", "0-2"])]
    tabla = sub.pivot_table(index="zona", columns="conteo", values="location_plus", aggfunc="mean")
    n = sub.pivot_table(index="zona", columns="conteo", values="location_plus", aggfunc="size")
    tabla = tabla.round(1).join(n.add_prefix("n_"))
    tabla.index = [f"{z} {NOMBRES_ZONA[z]}" for z in tabla.index]
    return tabla[["3-0", "0-2", "n_3-0", "n_0-2"]]


def estabilidad_entre_temporadas(scores, col):
    """Correlación del score medio de cada pitcher en 2024 vs 2025 (>= MIN_PITCHES cada año)."""
    por_año = scores.groupby(["pitcher_anon_id", "year"])[col].agg(["mean", "size"])
    por_año = por_año[por_año["size"] >= MIN_PITCHES]["mean"].unstack("year").dropna()
    años = sorted(por_año.columns)
    return por_año[años[0]].corr(por_año[años[1]]), len(por_año)


def main():
    datos = preparar_ubicacion(agregar_resultado(cargar_datos()))
    entrenamiento, _test = separar_test(datos)
    oof = pd.read_parquet(CARPETA_SALIDAS / "predicciones_oof_location.parquet")

    xrv = calcular_xrv(oof, "real")
    meta = entrenamiento[["PitchUID", "AutoPitchType", "year", "zona", "outcome_class"]]
    scores = xrv.merge(meta, on="PitchUID", how="left", validate="one_to_one")
    assert scores["year"].notna().all(), "hay pitches del OOF que no están en el entrenamiento"
    scores["location_plus"] = a_escala_plus(-scores["xrv_vs_conteo"], scores["year"])
    scores["conteo"] = scores["Balls"].astype(str) + "-" + scores["Strikes"].astype(str)

    # ---- salidas
    CARPETA_SALIDAS.mkdir(exist_ok=True)
    (scores[["PitchUID", "pitcher_anon_id", "AutoPitchType", "year", "Balls", "Strikes",
             "xrv", "xrv_vs_conteo", "location_plus"]]
     .rename(columns={"xrv": "xrv_location"})
     .to_parquet(CARPETA_SALIDAS / "scores_location_pitch.parquet", index=False))
    (scores.groupby(["Balls", "Strikes"], as_index=False)["xrv_conteo"].first()
     .to_csv(CARPETA_SALIDAS / "xrv_promedio_conteo_location.csv", index=False))
    por_tipo = (scores.groupby(["pitcher_anon_id", "AutoPitchType"])
                .agg(location_plus=("location_plus", "mean"), n=("location_plus", "size")).reset_index())
    por_tipo.to_csv(CARPETA_SALIDAS / "scores_location_pitcher_tipo.csv", index=False)
    por_pitcher = (scores.groupby("pitcher_anon_id")                  # ponderado por uso = media de sus pitches
                   .agg(location_plus=("location_plus", "mean"), n=("location_plus", "size")).reset_index())
    por_pitcher["location_plus_pitcher"] = a_escala_pitcher(por_pitcher["location_plus"], por_pitcher["n"])   # 100 ± 10 entre pitchers
    por_pitcher.to_csv(CARPETA_SALIDAS / "scores_location_pitcher.csv", index=False)

    # ---- Sanity 1: por construcción, media ~100 y sd ~10 en cada temporada
    print("Sanity 1 - Location+ por temporada (media ~100, sd ~10 por construcción):")
    print(scores.groupby("year")["location_plus"].agg(["mean", "std", "count"]).round(2).to_string())

    # ---- Sanity 2: restar el promedio deja cada conteo centrado en 0
    centro = scores.groupby("conteo")["xrv_vs_conteo"].mean().abs().max()
    print(f"\nSanity 2 - máx |media de xRV_vs_conteo| entre los 12 conteos: {centro:.1e} (debe ser ~0)")
    print("   xRV promedio por conteo (lo que se resta; + = bueno para el bateador):")
    print(scores.groupby("conteo")["xrv_conteo"].first().round(4).to_string())

    # ---- Sanity 3: la ubicación buena depende del conteo
    print("\nSanity 3 - Location+ medio por zona en 3-0 y en 0-2:")
    print(revisar_zona_por_conteo(scores).to_string())

    # ---- Sanity 4: a nivel pitcher, más Location+ debería ir con menos bolas
    scores["es_ball"] = (scores["outcome_class"] == "ball").astype(int)
    pb = (scores.groupby("pitcher_anon_id")
          .agg(loc=("location_plus", "mean"), ball=("es_ball", "mean"), n=("es_ball", "size")))
    pb = pb[pb["n"] >= MIN_PITCHES]
    print(f"\nSanity 4 - correlación (nivel pitcher, n>={MIN_PITCHES}: {len(pb)} pitchers) "
          f"Location+ vs tasa de bolas: r = {pb['loc'].corr(pb['ball']):.3f} (debe ser negativa)")

    # ---- Sanity 5: Location+ vs Stuff+ (miden cosas distintas) y estabilidad entre temporadas
    stuff, nombre_stuff = cargar_stuff_oficial()
    ambos = scores.merge(stuff, on="PitchUID", how="inner", validate="one_to_one")
    pp = (ambos.groupby("pitcher_anon_id")
          .agg(loc=("location_plus", "mean"), stuff=("stuff_plus", "mean"), n=("stuff_plus", "size")))
    pp = pp[pp["n"] >= MIN_PITCHES]
    print(f"\nSanity 5 - correlación Location+ vs {nombre_stuff} (nivel pitcher, {len(pp)} pitchers): "
          f"r = {pp['loc'].corr(pp['stuff']):.3f}")
    for col, nombre in [("location_plus", "Location+"), ("stuff_plus", nombre_stuff)]:
        r, n = estabilidad_entre_temporadas(ambos, col)
        print(f"   estabilidad {nombre} 2024 -> 2025 ({n} pitchers con >= {MIN_PITCHES} en cada año): r = {r:.3f}")

    print(f"\nGuardado en {CARPETA_SALIDAS}: scores_location_pitch.parquet, scores_location_pitcher_tipo.csv, "
          f"scores_location_pitcher.csv, xrv_promedio_conteo_location.csv")


if __name__ == "__main__":
    main()
