"""
Stuff+ v3: Stuff+ v2 (altitud) + features de arsenal (arsenal.py).

Mismos folds, parámetros y datos que v2 (modelo_stuff_altitud.py): la diferencia de log-loss entre v2 y v3
mide solo lo que aporta el arsenal. Se compara también fold a fold (comparación pareada, como en el Paso 6).

Salidas (fuera del repo):
    ../salidas/predicciones_oof_stuff_arsenal.parquet, validacion_stuff_arsenal.csv
    ../modelos/modelo_stuff_arsenal_fold{1..5}.txt + modelo_stuff_arsenal.txt

Uso:
    python modelo_stuff_arsenal.py      # antes: modelo_stuff_altitud.py
"""
import numpy as np
import pandas as pd

from arsenal import COLS_ARSENAL, agregar_arsenal
from modelo_multiclase import (CARPETA_MODELOS, CARPETA_SALIDAS, CONTEO, GRUPO, PARAMETROS,
                               entrenar_con_early_stopping, validacion_cruzada)
from modelo_stuff_altitud import FEATURES_V2, cargar_entrenamiento, leer_folds
from objetivo import CLASES

FEATURES_V3 = FEATURES_V2 + COLS_ARSENAL


def cargar_entrenamiento_arsenal():
    return agregar_arsenal(cargar_entrenamiento())


def main():
    df = cargar_entrenamiento_arsenal()
    print(f"Entrenamiento: {len(df):,} lanzamientos | sin recta primaria (arsenal NaN): "
          f"{100 * df['velo_diff_vs_fb'].isna().mean():.2f} %")
    print("Arsenal (medias): " + ", ".join(f"{c} {df[c].mean():+.2f}" for c in COLS_ARSENAL))
    print(f"Features: {FEATURES_V3}\n")

    fold_de_pitcher = leer_folds()
    resultados, pred_oof, modelos, por_clase, fold_de_fila = validacion_cruzada(
        df, PARAMETROS, features=FEATURES_V3, fold_de_pitcher=fold_de_pitcher)
    assert (fold_de_fila == df[GRUPO].map(fold_de_pitcher).to_numpy()).all()

    v2 = pd.read_csv(CARPETA_SALIDAS / "validacion_stuff_altitud.csv")
    dif = v2["logloss_modelo"].to_numpy() - resultados["logloss_modelo"].to_numpy()      # > 0: v3 mejor que v2
    print("\nResumen de la validación cruzada v3 (por pitcher, folds de Stuff+):")
    print(resultados.round(4).to_string(index=False))
    print(f"\nMejora sobre el baseline: v3 {resultados['mejora_%'].mean():.2f}% vs v2 {v2['mejora_%'].mean():.2f}%")
    print(f"Comparación pareada v2 − v3 (log-loss; + = v3 mejor): media {dif.mean():+.5f}, std {dif.std(ddof=1):.5f}, "
          f"v3 gana en {int((dif > 0).sum())}/5 folds")
    print("\nMejora de log-loss binaria por clase (v3 vs baseline por conteo):")
    print(por_clase.to_string(index=False))

    CARPETA_SALIDAS.mkdir(exist_ok=True)
    salida = df[["PitchUID", GRUPO] + CONTEO + ["outcome_class"]].copy()
    for i, clase in enumerate(CLASES):
        salida[f"p_{clase}"] = pred_oof[:, i]
    salida.to_parquet(CARPETA_SALIDAS / "predicciones_oof_stuff_arsenal.parquet", index=False)
    resultados.to_csv(CARPETA_SALIDAS / "validacion_stuff_arsenal.csv", index=False)
    CARPETA_MODELOS.mkdir(exist_ok=True)
    for parte, modelo in enumerate(modelos, start=1):
        modelo.booster_.save_model(str(CARPETA_MODELOS / f"modelo_stuff_arsenal_fold{parte}.txt"),
                                   num_iteration=modelo.best_iteration_)
    final = entrenar_con_early_stopping(df[FEATURES_V3], df["outcome_id"], df[GRUPO], PARAMETROS)
    final.booster_.save_model(str(CARPETA_MODELOS / "modelo_stuff_arsenal.txt"), num_iteration=final.best_iteration_)
    print(f"\nGuardado: predicciones_oof_stuff_arsenal.parquet, validacion_stuff_arsenal.csv y los modelos (v3)")


if __name__ == "__main__":
    main()
