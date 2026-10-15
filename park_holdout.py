"""
Altitud, Idea 5: park holdout. ¿El modelo funciona en un estadio que NO vio al entrenar?

Dos pruebas, con los mismos parámetros que Stuff+ y solo con las temporadas de entrenamiento:
  A. Sin Harp Helú: entrena con todos los juegos menos los de Harp Helú y prueba en Harp Helú.
  B. Sin altura extrema: entrena sin los estadios de altura extrema (que incluyen a Harp Helú) y prueba ahí.
     Es la prueba dura: la densidad de entrenamiento no llega a los valores de la prueba.

Tres modelos (misma física, mismos pitchers):
  v1       física con el movimiento medido, sin densidad                      (Stuff+ v1)
  v2_mov   movimiento normalizado a nivel del mar, sin densidad               (solo canal del movimiento)
  v2       movimiento normalizado + densidad_juego                            (Stuff+ v2)

Se mide en los juegos de prueba: log-loss contra el baseline por conteo (frecuencias del entrenamiento) y el
sesgo (probabilidad predicha − frecuencia real) de ball, whiff, foul e in_play.

Los pitchers sí se repiten entre entrenamiento y prueba (es una prueba de ESTADIO, no de pitcher; la de pitcher
ya la hace la validación cruzada por pitcher).

Salida (fuera del repo): ../salidas/park_holdout.csv

Uso:
    python park_holdout.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from efecto_altitud import RUTA_ESTADIOS
from modelo_multiclase import (CARPETA_SALIDAS, CONTEO, FEATURES, GRUPO, PARAMETROS, baseline_por_conteo,
                               entrenar_con_early_stopping)
from modelo_stuff_altitud import FEATURES_V2, cargar_entrenamiento
from objetivo import CLASES

MODELOS = {
    "v1 (sin altitud)": FEATURES,
    "v2_mov (movimiento normalizado)": [f for f in FEATURES_V2 if f != "densidad_juego"],
    "v2 (movimiento normalizado + densidad)": FEATURES_V2,
}


def evaluar(df, test_mask, nombre_prueba, filas):
    train, test = df[~test_mask], df[test_mask]
    y_tr, y_te = train["outcome_id"], test["outcome_id"].to_numpy()
    p_base = baseline_por_conteo(y_tr, train[CONTEO], test[CONTEO])
    ll_base = log_loss(y_te, p_base, labels=range(5))
    print(f"\n{nombre_prueba}: entrena {len(train):,} pitches | prueba {len(test):,} pitches "
          f"(densidad de prueba media {test['densidad_juego'].mean():.3f} vs entrenamiento {train['densidad_juego'].mean():.3f})")
    for nombre, features in MODELOS.items():
        modelo = entrenar_con_early_stopping(train[features], y_tr, train[GRUPO], PARAMETROS)
        p = modelo.predict_proba(test[features])
        ll = log_loss(y_te, p, labels=range(5))
        fila = {"prueba": nombre_prueba, "modelo": nombre, "n_prueba": len(test), "logloss": ll, "logloss_baseline": ll_base,
                "mejora_%": 100 * (ll_base - ll) / ll_base}
        for clase in ["ball", "whiff", "foul", "in_play"]:
            i = CLASES.index(clase)
            fila[f"sesgo_{clase}_pp"] = 100 * (p[:, i].mean() - (y_te == i).mean())
        filas.append(fila)
        print(f"   {nombre:40s} log-loss {ll:.4f} (baseline {ll_base:.4f}) mejora {fila['mejora_%']:.2f}% | sesgo pp: "
              + ", ".join(f"{c} {fila[f'sesgo_{c}_pp']:+.2f}" for c in ["ball", "whiff", "foul", "in_play"]))


def main():
    df = cargar_entrenamiento()
    est = pd.read_parquet(RUTA_ESTADIOS)[["game_anon_id", "altitud"]]
    df = df.merge(est, on="game_anon_id", how="left", validate="many_to_one")
    extrema = (df["altitud"] == "Extreme Altitude").to_numpy()
    print(f"Pitches: {len(df):,} | en Harp Helú: {int(df['harp_helu'].sum()):,} | en altura extrema: {int(extrema.sum()):,}")

    filas = []
    evaluar(df, df["harp_helu"].to_numpy(), "A. sin Harp Helú", filas)
    evaluar(df, extrema, "B. sin altura extrema", filas)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_SALIDAS / "park_holdout.csv", index=False)
    print("\nGuardado: ../salidas/park_holdout.csv")


if __name__ == "__main__":
    main()
