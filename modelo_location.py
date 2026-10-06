"""
Location+: modelo multiclase de resultado del lanzamiento según DÓNDE se puso.

Pregunta: ¿la puso donde debía? Mismo método que Stuff+ (modelo_multiclase.py):
LightGBM multiclase con las 5 clases de objetivo.py, validación GroupKFold por pitcher,
early stopping, baseline por conteo y predicciones out-of-fold. Cambian las features y
min_child_samples (2000, ver PARAMETROS_LOCATION):

    ubicación:          PlateLocHeight, lado_relativo (positivo = adentro, ver zona.py)
    conteo:             Balls, Strikes
    lado del bateador:  bateador_zurdo, misma_mano (las dos: juntas distinguen los
                        4 emparejamientos pitcher/bateador)
    tipo de pitch:      tipo_pitch (categórica, de AutoPitchType)

No ve la física (velocidad, movimiento, spin): eso es Stuff+. Pitching+ ve todo.

Partición: se reutiliza EXACTAMENTE la de Stuff+ (../salidas/folds.csv, escrito por
modelo_multiclase.py), así cada pitcher cae en el mismo fold de prueba en los tres
modelos y sus predicciones out-of-fold son comparables.

Qué hace, en orden:
  1. Carga y limpia, agrega el target, prepara la ubicación (zona.py; quita los pitches
     sin lado de bateador conocido) y se queda SOLO con el entrenamiento.
  2. Validación cruzada con los folds de Stuff+; compara contra el baseline por conteo.
  3. Calibración general, por conteo y por zona (1-9 + fuera).
  4. Guarda predicciones out-of-fold, los 5 modelos de los folds y un modelo final.

Uso:
    python modelo_location.py      # antes: python modelo_multiclase.py (escribe folds.csv)
"""
import numpy as np
import pandas as pd

from limpieza import cargar_datos
from modelo_multiclase import (CARPETA_MODELOS, CARPETA_SALIDAS, CONTEO, GRUPO, PARAMETROS,
                               entrenar_con_early_stopping, revisar_calibracion,
                               revisar_calibracion_por_conteo, validacion_cruzada)
from objetivo import CLASES, agregar_resultado
from particion import separar_test
from zona import NOMBRES_ZONA, preparar_ubicacion

# ---------------------------------------------------------------- configuración
UBICACION = ["PlateLocHeight", "lado_relativo"]
BATEADOR = ["bateador_zurdo", "misma_mano"]
FEATURES_LOCATION = UBICACION + CONTEO + BATEADOR + ["tipo_pitch"]
# Igual que Stuff+ salvo min_child_samples, elegido por CV pareada (busqueda_min_child_location.py):
# 1000 tiene la menor log-loss pero 2000 es indistinguible (1000 gana solo 3/5 folds) y se
# prefiere el más regularizado; 5000 pierde en 5/5 folds contra 2000.
PARAMETROS_LOCATION = {**PARAMETROS, "min_child_samples": 2000}
RUTA_FOLDS = CARPETA_SALIDAS / "folds.csv"


def leer_folds():
    """Series pitcher -> fold de prueba, la misma partición que usó Stuff+."""
    if not RUTA_FOLDS.exists():
        raise FileNotFoundError(f"No existe {RUTA_FOLDS}: corre antes python modelo_multiclase.py")
    folds = pd.read_csv(RUTA_FOLDS)
    return folds.set_index(GRUPO)["fold"]


def revisar_calibracion_por_zona(df, pred_oof):
    """Por zona (1-9 y fuera): prob. predicha vs frecuencia real de ball y called_strike."""
    outcome = df["outcome_id"].to_numpy()
    filas = []
    for z, posiciones in df.groupby("zona").groups.items():
        pos = np.asarray(posiciones)
        fila = {"zona": f"{z} {NOMBRES_ZONA[z]}", "n": len(pos)}
        for clase in ["ball", "called_strike", "whiff"]:
            i = CLASES.index(clase)
            fila[f"{clase}_real_%"] = round(100 * (outcome[pos] == i).mean(), 2)
            fila[f"{clase}_pred_%"] = round(100 * pred_oof[pos, i].mean(), 2)
        filas.append(fila)
    return pd.DataFrame(filas)


def main():
    df = preparar_ubicacion(agregar_resultado(cargar_datos()))
    entrenamiento, _test = separar_test(df)
    df = entrenamiento.reset_index(drop=True)          # el test sellado no se toca
    print(f"Entrenamiento: {len(df):,} lanzamientos | {df[GRUPO].nunique():,} pitchers "
          f"(test sellado apartado: {len(_test):,} lanzamientos)")
    print(f"Features: {FEATURES_LOCATION}\n")

    fold_de_pitcher = leer_folds()
    resultados, pred_oof, modelos, por_clase, fold_de_fila = validacion_cruzada(
        df, PARAMETROS_LOCATION, features=FEATURES_LOCATION, fold_de_pitcher=fold_de_pitcher)
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

    print("\nCalibración por zona:")
    print(revisar_calibracion_por_zona(df, pred_oof).to_string(index=False))

    # ---- guardar predicciones fuera de muestra (mismas columnas que predicciones_oof.parquet)
    CARPETA_SALIDAS.mkdir(exist_ok=True)
    salida = df[["PitchUID", GRUPO] + CONTEO + ["outcome_class"]].copy()
    for i, clase in enumerate(CLASES):
        salida[f"p_{clase}"] = pred_oof[:, i]
    salida.to_parquet(CARPETA_SALIDAS / "predicciones_oof_location.parquet", index=False)
    resultados.to_csv(CARPETA_SALIDAS / "validacion_location.csv", index=False)

    # ---- guardar los 5 modelos de los folds
    CARPETA_MODELOS.mkdir(exist_ok=True)
    for parte, modelo in enumerate(modelos, start=1):
        modelo.booster_.save_model(str(CARPETA_MODELOS / f"modelo_location_fold{parte}.txt"),
                                   num_iteration=modelo.best_iteration_)

    # ---- modelo final: entrenado con TODO el entrenamiento (para el dashboard y pitches nuevos)
    modelo_final = entrenar_con_early_stopping(df[FEATURES_LOCATION], df["outcome_id"], df[GRUPO],
                                               PARAMETROS_LOCATION)
    modelo_final.booster_.save_model(str(CARPETA_MODELOS / "modelo_location.txt"),
                                     num_iteration=modelo_final.best_iteration_)

    print(f"\nGuardado: {CARPETA_SALIDAS / 'predicciones_oof_location.parquet'} ({len(salida):,} filas)")
    print(f"Guardado: {CARPETA_SALIDAS / 'validacion_location.csv'}")
    print(f"Guardado: {CARPETA_MODELOS / 'modelo_location_fold{1..5}.txt'} (5 folds)")
    print(f"Guardado: {CARPETA_MODELOS / 'modelo_location.txt'} (final)")


if __name__ == "__main__":
    main()
