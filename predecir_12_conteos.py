"""
Paso 8 (insumo): predicciones en los 12 conteos (neutralización del conteo).

Stuff+ es neutral al conteo: el mismo pitch se evalúa en los 12 conteos y se
promedia con los pesos de la liga. Aquí producimos esa pieza: para cada pitch,
con la FÍSICA fija, variamos (Balls, Strikes) sobre los 12 conteos y predecimos
P(k | física, conteo). No calculamos el xRV todavía (eso necesita RV_k(conteo)
de run_values.py y los pesos de la liga; lo hará scoring.py).

Honestidad (out-of-fold): cada pitch de entrenamiento se predice con el modelo
del fold que NO vio a su pitcher, leyendo la asignación de ../salidas/folds.csv
(escrita por modelo_multiclase.py) y los 5 modelos de fold. Así no se reconstruye
la partición: si el orden o los datos cambiaran, un pitcher no se asignaría mal
en silencio.

Salida: ../salidas/predicciones_12conteos.parquet (formato largo, probs float32):
    PitchUID, pitcher_anon_id, Balls, Strikes, p_ball, ..., p_in_play

Uso:
    python predecir_12_conteos.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from limpieza import cargar_datos
from objetivo import CLASES, agregar_resultado
from particion import separar_test
from modelo_multiclase import (FISICA, FEATURES, GRUPO, N_PARTES,
                               CARPETA_SALIDAS, CARPETA_MODELOS)

CONTEOS = [(b, s) for b in (0, 1, 2, 3) for s in (0, 1, 2)]   # los 12 conteos
COLS_PROB = [f"p_{c}" for c in CLASES]


def predecir_12_conteos(df, modelo):
    """Para cada fila de df predice P(k) en los 12 conteos, con la física fija.

    `modelo`: un LGBMClassifier (predict_proba) o un Booster de LightGBM (predict).
    Devuelve un DataFrame largo (una fila por pitch × conteo) con los IDs que
    traiga df (`PitchUID`, `pitcher_anon_id`), `Balls`, `Strikes` y `p_<clase>`.
    """
    predecir = modelo.predict_proba if hasattr(modelo, "predict_proba") else modelo.predict
    df = df.reset_index(drop=True)
    base = df[[c for c in ["PitchUID", GRUPO] if c in df.columns]]
    fisica = df[FISICA]

    partes = []
    for b, s in CONTEOS:
        X = fisica.copy()
        X["Balls"], X["Strikes"] = float(b), float(s)
        p = np.asarray(predecir(X[FEATURES]), dtype=np.float32)
        bloque = base.copy()
        bloque["Balls"] = np.int8(b)
        bloque["Strikes"] = np.int8(s)
        bloque[COLS_PROB] = p
        partes.append(bloque)
    return pd.concat(partes, ignore_index=True)


def predecir_oof_entrenamiento():
    """Predice los 12 conteos de cada pitch de entrenamiento con su modelo OOF."""
    df = agregar_resultado(cargar_datos()).reset_index(drop=True)
    entrenamiento, _test = separar_test(df)
    df = entrenamiento.reset_index(drop=True)

    folds = pd.read_csv(CARPETA_SALIDAS / "folds.csv")          # pitcher_anon_id, fold
    fold_de_fila = df[GRUPO].map(dict(zip(folds[GRUPO], folds["fold"])))
    if fold_de_fila.isna().any():
        faltan = df.loc[fold_de_fila.isna(), GRUPO].nunique()
        raise SystemExit(f"{faltan} pitchers no están en folds.csv; vuelve a correr modelo_multiclase.py.")

    modelos = {k: lgb.Booster(model_file=str(CARPETA_MODELOS / f"modelo_resultado_fold{k}.txt"))
               for k in range(1, N_PARTES + 1)}

    partes = []
    for k in range(1, N_PARTES + 1):
        sub = df[fold_de_fila == k]
        print(f"  fold {k}: {len(sub):,} pitches × 12 conteos")
        partes.append(predecir_12_conteos(sub, modelos[k]))
    return df, pd.concat(partes, ignore_index=True)


def main():
    print("Prediciendo los 12 conteos (out-of-fold) sobre el entrenamiento...")
    df_entr, pred12 = predecir_oof_entrenamiento()
    pred12[COLS_PROB] = pred12[COLS_PROB].astype("float32")

    CARPETA_SALIDAS.mkdir(exist_ok=True)
    salida = CARPETA_SALIDAS / "predicciones_12conteos.parquet"
    pred12.to_parquet(salida, index=False)
    print(f"\nGuardado: {salida}  ({len(pred12):,} filas = {len(df_entr):,} pitches × 12 conteos)")

    # ---- Sanity 1: las 5 probabilidades suman 1
    suma = pred12[COLS_PROB].sum(axis=1)
    print(f"\nSanity 1 - suma de las 5 probabilidades: max |suma - 1| = {np.abs(suma - 1).max():.2e}")

    # ---- Sanity 2: coincidir con predicciones_oof.parquet en el conteo REAL de cada pitch
    oof = pd.read_parquet(CARPETA_SALIDAS / "predicciones_oof.parquet")
    oof["Balls"] = oof["Balls"].astype("int8")
    oof["Strikes"] = oof["Strikes"].astype("int8")
    clave = ["PitchUID", "Balls", "Strikes"]
    m = oof[clave + COLS_PROB].merge(pred12[clave + COLS_PROB], on=clave, suffixes=("_oof", "_12"))
    dif = np.abs(m[[c + "_oof" for c in COLS_PROB]].to_numpy()
                 - m[[c + "_12" for c in COLS_PROB]].to_numpy())
    print(f"Sanity 2 — vs predicciones_oof en el conteo real: {len(m):,} pitches casados "
          f"(de {len(oof):,}); diferencia máxima absoluta = {dif.max():.2e}")

    # ---- Sanity 3a: probabilidad promedio por conteo (12 × 5)
    tabla = (pred12.groupby(["Balls", "Strikes"])[COLS_PROB].mean() * 100).round(2)
    print("\nSanity 3a — probabilidad promedio por conteo (%, 12 × 5):")
    print(tabla.to_string())

    # ---- Sanity 3b: four-seam y slider medianos en 0-0, 0-2, 3-0, 3-2 (modelo final)
    final = lgb.Booster(model_file=str(CARPETA_MODELOS / "modelo_resultado.txt"))
    ejemplos = []
    for tipo in ("Four-Seam", "Slider"):
        med = df_entr.loc[df_entr["AutoPitchType"] == tipo, FISICA].median().to_frame().T
        med["PitchUID"] = tipo
        ejemplos.append(med)
    pe = predecir_12_conteos(pd.concat(ejemplos, ignore_index=True), final)
    cuatro = {(0, 0), (0, 2), (3, 0), (3, 2)}
    pe = pe[[(b, s) in cuatro for b, s in zip(pe["Balls"], pe["Strikes"])]].copy()
    pe[COLS_PROB] = (pe[COLS_PROB] * 100).round(2)
    pe = pe.rename(columns={"PitchUID": "pitch"}).sort_values(["pitch", "Balls", "Strikes"])
    print("\nSanity 3b — four-seam y slider medianos (prob %, modelo final):")
    print(pe.to_string(index=False))


if __name__ == "__main__":
    main()
