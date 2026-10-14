"""
Pitching+: modelo multiclase de resultado del lanzamiento viendo TODO.

Pregunta: ¿qué tan bueno fue este lanzamiento concreto? A diferencia de Stuff+
(solo física) y Location+ (solo ubicación), Pitching+ ve física + ubicación +
conteo + lado del bateador. Mismo método que los otros dos (modelo_multiclase.py):
LightGBM multiclase con las 5 clases de objetivo.py, GroupKFold por pitcher,
early stopping, baseline por conteo y predicciones out-of-fold.

Features (sin altitud por ahora; FEATURES_PITCHING es una lista para añadir
`densidad_juego` después):
    física (8):     RelSpeed, SpinRate, InducedVertBreak, HorzBreak, Extension,
                    RelHeight, RelSide, SpinAxis   (se reusa FISICA de Stuff+)
    conteo:         Balls, Strikes
    ubicación:      PlateLocHeight, lado_relativo (positivo = adentro, ver zona.py)
    lado bateador:  bateador_zurdo, misma_mano (juntas distinguen los 4 emparejamientos)

min_child_samples = 1000 (el mismo que Stuff+).

Partición: se reutiliza EXACTAMENTE la de Stuff+/Location+ (../salidas/folds.csv),
así cada pitcher cae en el mismo fold en los tres modelos y los OOF son comparables.

Este archivo entrena el modelo y guarda el OOF; el *score* Pitching+ (xRV en el
conteo real menos el promedio del conteo, con scoring.py contexto "real") es aparte.

Uso:
    python pitching_plus.py      # antes: python modelo_multiclase.py (escribe folds.csv)
"""
import pandas as pd

from limpieza import cargar_datos
from modelo_multiclase import (CARPETA_MODELOS, CARPETA_SALIDAS, CONTEO, FISICA, GRUPO, PARAMETROS,
                               entrenar_con_early_stopping, revisar_calibracion,
                               revisar_calibracion_por_conteo, validacion_cruzada)
from objetivo import CLASES, agregar_resultado
from particion import separar_test
from zona import preparar_ubicacion

# ---------------------------------------------------------------- configuración
UBICACION = ["PlateLocHeight", "lado_relativo"]
LADO = ["bateador_zurdo", "misma_mano"]
# Para altitud v2: FEATURES_PITCHING = FISICA + CONTEO + UBICACION + LADO + ["densidad_juego"]
FEATURES_PITCHING = FISICA + CONTEO + UBICACION + LADO
PARAMETROS_PITCHING = {**PARAMETROS, "min_child_samples": 1000}   # igual que Stuff+
RUTA_FOLDS = CARPETA_SALIDAS / "folds.csv"


def leer_folds():
    """Series pitcher -> fold de prueba, la misma partición que usó Stuff+."""
    if not RUTA_FOLDS.exists():
        raise FileNotFoundError(f"No existe {RUTA_FOLDS}: corre antes python modelo_multiclase.py")
    return pd.read_csv(RUTA_FOLDS).set_index(GRUPO)["fold"]


def main():
    df = preparar_ubicacion(agregar_resultado(cargar_datos()))
    entrenamiento, _test = separar_test(df)
    df = entrenamiento.reset_index(drop=True)          # el test sellado no se toca
    print(f"Entrenamiento: {len(df):,} lanzamientos | {df[GRUPO].nunique():,} pitchers "
          f"(test sellado apartado: {len(_test):,} lanzamientos)")
    print(f"Features: {FEATURES_PITCHING}\n")

    fold_de_pitcher = leer_folds()
    resultados, pred_oof, modelos, por_clase, fold_de_fila = validacion_cruzada(
        df, PARAMETROS_PITCHING, features=FEATURES_PITCHING, fold_de_pitcher=fold_de_pitcher)
    # sanity: cada fila cayó en el fold de su pitcher según folds.csv
    assert (fold_de_fila == df[GRUPO].map(fold_de_pitcher).to_numpy()).all()

    print("\nResumen de la validación cruzada (por pitcher, folds de Stuff+):")
    print(resultados.round(4).to_string(index=False))
    print(f"\nMejora promedio sobre el baseline: {resultados['mejora_%'].mean():.2f}% "
          f"(peor parte {resultados['mejora_%'].min():.2f}%, mejor {resultados['mejora_%'].max():.2f}%)")

    print("\nMejora de log-loss binaria por clase (esa clase vs. las demás; modelo vs baseline por conteo):")
    print(por_clase.to_string(index=False))

    print("\nCalibración general:")
    print(revisar_calibracion(df, pred_oof).to_string(index=False))

    calib_conteo = revisar_calibracion_por_conteo(df, pred_oof)
    print(f"\nCalibración por conteo: máx |dif| = {calib_conteo['dif_pp'].abs().max():.2f} pp")
    grandes = calib_conteo.loc[calib_conteo["dif_pp"].abs() >= 1]
    print(grandes.to_string(index=False) if len(grandes) else "(ninguna celda conteo × clase con |dif| >= 1 pp)")

    # ---- guardar predicciones fuera de muestra (mismas columnas que predicciones_oof.parquet)
    CARPETA_SALIDAS.mkdir(exist_ok=True)
    salida = df[["PitchUID", GRUPO] + CONTEO + ["outcome_class"]].copy()
    for i, clase in enumerate(CLASES):
        salida[f"p_{clase}"] = pred_oof[:, i]
    salida.to_parquet(CARPETA_SALIDAS / "predicciones_oof_pitching.parquet", index=False)
    resultados.to_csv(CARPETA_SALIDAS / "validacion_pitching.csv", index=False)

    # ---- los 5 modelos de los folds + el modelo final (entrenado con todo el entrenamiento)
    CARPETA_MODELOS.mkdir(exist_ok=True)
    for parte, modelo in enumerate(modelos, start=1):
        modelo.booster_.save_model(str(CARPETA_MODELOS / f"modelo_pitching_fold{parte}.txt"),
                                   num_iteration=modelo.best_iteration_)
    modelo_final = entrenar_con_early_stopping(df[FEATURES_PITCHING], df["outcome_id"], df[GRUPO],
                                               PARAMETROS_PITCHING)
    modelo_final.booster_.save_model(str(CARPETA_MODELOS / "modelo_pitching.txt"),
                                     num_iteration=modelo_final.best_iteration_)

    print(f"\nGuardado: {CARPETA_SALIDAS / 'predicciones_oof_pitching.parquet'} ({len(salida):,} filas)")
    print(f"Guardado: {CARPETA_SALIDAS / 'validacion_pitching.csv'}")
    print(f"Guardado: {CARPETA_MODELOS / 'modelo_pitching_fold{1..5}.txt'} (5 folds)")
    print(f"Guardado: {CARPETA_MODELOS / 'modelo_pitching.txt'} (final)")


if __name__ == "__main__":
    main()
