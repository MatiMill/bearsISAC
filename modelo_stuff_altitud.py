"""
Stuff+ v2 (con altitud): modelo multiclase de resultado con el movimiento normalizado y la densidad.

Dos canales de la altitud, cada uno por su lado (docs/guia_proyecto.md §7 paso 10):
  1. Movimiento: IVB y HB se llevan al aire de nivel del mar (IVB_n, HB_n, altitud_movimiento.py,
     con el efecto medido en efecto_altitud.py). Así el modelo califica la CALIDAD del pitch y no el
     estadio en el que se midió (en v1 los pitches de Harp Helú salían con Stuff+ ≈ 97 vs ≈ 100 fuera).
  2. Resultado: `densidad_juego` entra como feature, para que el modelo aprenda cómo cambian las
     probabilidades de whiff, foul y bola en juego con el aire (p. ej. batazos que vuelan más).

Features (v1 + altitud):
    física (8):  RelSpeed, SpinRate, IVB_n, HB_n, Extension, RelHeight, RelSide, SpinAxis
    contexto:    Balls, Strikes, densidad_juego

Mismos folds que Stuff+ v1 (../salidas/folds.csv) y mismos parámetros (min_child_samples = 1000),
para poder comparar log-loss pitch a pitch contra v1.

Salidas (fuera del repo):
    ../salidas/predicciones_oof_stuff_altitud.parquet   (mismas columnas que predicciones_oof.parquet)
    ../salidas/validacion_stuff_altitud.csv
    ../modelos/modelo_stuff_altitud_fold{1..5}.txt + modelo_stuff_altitud.txt

Uso:
    python modelo_stuff_altitud.py      # antes: python modelo_multiclase.py (escribe folds.csv)
"""
import numpy as np
import pandas as pd

from altitud_movimiento import normalizar_movimiento
from efecto_altitud import agregar_densidad
from limpieza import cargar_datos
from modelo_multiclase import (CARPETA_MODELOS, CARPETA_SALIDAS, CONTEO, GRUPO, PARAMETROS,
                               entrenar_con_early_stopping, revisar_calibracion,
                               revisar_calibracion_por_conteo, validacion_cruzada)
from objetivo import CLASES, agregar_resultado
from particion import separar_test

FISICA_V2 = ["RelSpeed", "SpinRate", "IVB_n", "HB_n", "Extension", "RelHeight", "RelSide", "SpinAxis"]
FEATURES_V2 = FISICA_V2 + CONTEO + ["densidad_juego"]
RUTA_FOLDS = CARPETA_SALIDAS / "folds.csv"


def leer_folds():
    """Series pitcher -> fold de prueba, la misma partición que usó Stuff+ v1."""
    if not RUTA_FOLDS.exists():
        raise FileNotFoundError(f"No existe {RUTA_FOLDS}: corre antes python modelo_multiclase.py")
    return pd.read_csv(RUTA_FOLDS).set_index(GRUPO)["fold"]


def preparar(df):
    """Agrega densidad_juego, harp_helu, IVB_n y HB_n a los pitches limpios de df."""
    return normalizar_movimiento(agregar_densidad(df))


def cargar_entrenamiento():
    entrenamiento = separar_test(agregar_resultado(cargar_datos()))[0]   # el test sellado no se toca
    return preparar(entrenamiento).reset_index(drop=True)


def calibracion_por_aire(df, pred_oof):
    """Predicho vs real de whiff, foul e in_play en Harp Helú y en el resto (el aire importa)."""
    outcome = df["outcome_id"].to_numpy()
    filas = []
    for nombre, mascara in [("Harp Helú", df["harp_helu"].to_numpy()), ("resto", ~df["harp_helu"].to_numpy())]:
        fila = {"juegos": nombre, "n": int(mascara.sum())}
        for clase in ["ball", "whiff", "foul", "in_play"]:
            i = CLASES.index(clase)
            fila[f"{clase}_real_%"] = round(100 * (outcome[mascara] == i).mean(), 2)
            fila[f"{clase}_pred_%"] = round(100 * pred_oof[mascara, i].mean(), 2)
        filas.append(fila)
    return pd.DataFrame(filas)


def main():
    df = cargar_entrenamiento()
    print(f"Entrenamiento: {len(df):,} lanzamientos | {df[GRUPO].nunique():,} pitchers | "
          f"pitches en Harp Helú: {int(df['harp_helu'].sum()):,}")
    print(f"Features: {FEATURES_V2}\n")

    fold_de_pitcher = leer_folds()
    resultados, pred_oof, modelos, por_clase, fold_de_fila = validacion_cruzada(
        df, PARAMETROS, features=FEATURES_V2, fold_de_pitcher=fold_de_pitcher)
    assert (fold_de_fila == df[GRUPO].map(fold_de_pitcher).to_numpy()).all()

    print("\nResumen de la validación cruzada (por pitcher, folds de Stuff+):")
    print(resultados.round(4).to_string(index=False))
    print(f"\nMejora promedio sobre el baseline: {resultados['mejora_%'].mean():.2f}% "
          f"(peor parte {resultados['mejora_%'].min():.2f}%, mejor {resultados['mejora_%'].max():.2f}%)")
    v1 = pd.read_csv(CARPETA_SALIDAS / "validacion_multiclase.csv")
    print(f"   Stuff+ v1 (sin altitud): {v1['mejora_%'].mean():.2f}% | log-loss v1 {v1['logloss_modelo'].mean():.5f} "
          f"vs v2 {resultados['logloss_modelo'].mean():.5f} (menos es mejor)")

    print("\nMejora de log-loss binaria por clase (modelo vs baseline por conteo):")
    print(por_clase.to_string(index=False))
    print("\nCalibración general:")
    print(revisar_calibracion(df, pred_oof).to_string(index=False))
    calib = revisar_calibracion_por_conteo(df, pred_oof)
    print(f"\nCalibración por conteo: máx |dif| = {calib['dif_pp'].abs().max():.2f} pp")
    print("\nCalibración por aire (Harp Helú vs resto):")
    print(calibracion_por_aire(df, pred_oof).to_string(index=False))

    CARPETA_SALIDAS.mkdir(exist_ok=True)
    salida = df[["PitchUID", GRUPO] + CONTEO + ["outcome_class"]].copy()
    for i, clase in enumerate(CLASES):
        salida[f"p_{clase}"] = pred_oof[:, i]
    salida.to_parquet(CARPETA_SALIDAS / "predicciones_oof_stuff_altitud.parquet", index=False)
    resultados.to_csv(CARPETA_SALIDAS / "validacion_stuff_altitud.csv", index=False)

    CARPETA_MODELOS.mkdir(exist_ok=True)
    for parte, modelo in enumerate(modelos, start=1):
        modelo.booster_.save_model(str(CARPETA_MODELOS / f"modelo_stuff_altitud_fold{parte}.txt"),
                                   num_iteration=modelo.best_iteration_)
    final = entrenar_con_early_stopping(df[FEATURES_V2], df["outcome_id"], df[GRUPO], PARAMETROS)
    final.booster_.save_model(str(CARPETA_MODELOS / "modelo_stuff_altitud.txt"), num_iteration=final.best_iteration_)

    print(f"\nGuardado: {CARPETA_SALIDAS / 'predicciones_oof_stuff_altitud.parquet'} ({len(salida):,} filas)")
    print(f"Guardado: {CARPETA_MODELOS / 'modelo_stuff_altitud_fold{1..5}.txt'} y modelo_stuff_altitud.txt")


if __name__ == "__main__":
    main()
