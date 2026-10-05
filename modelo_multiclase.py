"""
Tarea 6: modelo multiclase de resultado del lanzamiento.

Para cada lanzamiento predice 5 probabilidades que suman 1:
    P(ball), P(called_strike), P(whiff), P(foul), P(in_play)
usando SOLO la física del pitch (Stuff+ puro: sin conteo ni ubicación).

Qué hace, en orden:
  1. Carga y limpia los datos, y agrega el target (objetivo.py).
  2. Validación cruzada de 5 partes agrupada por pitcher (GroupKFold):
     cada pitcher cae en la parte de prueba exactamente una vez y nunca
     está en entrenamiento y prueba al mismo tiempo.
  3. En cada parte compara la log-loss del modelo contra un baseline
     que siempre predice la frecuencia de cada clase.
  4. Revisa la calibración: probabilidad promedio predicha vs. frecuencia real.
  5. Guarda las predicciones "fuera de muestra" de cada lanzamiento
     (las usará la Tarea 9 para calcular xRV) y un modelo final entrenado con todo.

Uso:
    python modelo_multiclase.py
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from limpieza import cargar_datos
from objetivo import CLASES, agregar_resultado

# ---------------------------------------------------------------- configuración
FEATURES = ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak",
            "Extension", "RelHeight", "RelSide"]
GRUPO = "pitcher_anon_id"      # la validación se agrupa por esta columna
N_PARTES = 5                   # partes de la validación cruzada
SEMILLA = 42

# Los resultados se guardan FUERA del repo (junto al parquet), porque contienen datos del ISAC.
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_SALIDAS = RAIZ / "salidas"
CARPETA_MODELOS = RAIZ / "modelos"

PARAMETROS = dict(
    objective="multiclass",    # 5 clases con softmax
    n_estimators=2000,         # máximo de árboles; early stopping decide cuántos se usan
    learning_rate=0.05,        # cuánto aporta cada árbol
    num_leaves=31,             # tamaño máximo de cada árbol
    min_child_samples=200,     # mínimo de lanzamientos por hoja: evita memorizar ruido
    subsample=0.8,             # cada árbol ve el 80% de las filas...
    subsample_freq=1,
    colsample_bytree=0.8,      # ...y el 80% de las columnas
    random_state=SEMILLA,
    verbose=-1,
)


def entrenar_con_early_stopping(X, y, grupos):
    """Entrena un modelo apartando el 10% de los pitchers para decidir cuándo parar.

    Early stopping: después de cada árbol se mide la log-loss en esos pitchers
    apartados. Cuando deja de bajar durante 50 árboles seguidos, se para y se
    usa el número de árboles que dio la mejor log-loss.
    """
    corte = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEMILLA)
    idx_fit, idx_val = next(corte.split(X, y, groups=grupos))

    modelo = lgb.LGBMClassifier(**PARAMETROS)
    modelo.fit(
        X.iloc[idx_fit], y.iloc[idx_fit],
        eval_set=[(X.iloc[idx_val], y.iloc[idx_val])],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return modelo


def validacion_cruzada(df):
    """Devuelve (tabla de resultados por parte, matriz de predicciones fuera de muestra)."""
    X, y, grupos = df[FEATURES], df["outcome_id"], df[GRUPO]
    pred_oof = np.full((len(df), len(CLASES)), np.nan)
    filas = []

    for parte, (idx_train, idx_test) in enumerate(GroupKFold(n_splits=N_PARTES).split(X, y, grupos), start=1):
        modelo = entrenar_con_early_stopping(X.iloc[idx_train], y.iloc[idx_train], grupos.iloc[idx_train])
        assert list(modelo.classes_) == list(range(len(CLASES))), "falta alguna clase en el entrenamiento"

        p_test = modelo.predict_proba(X.iloc[idx_test])
        pred_oof[idx_test] = p_test

        # baseline: predecir siempre la frecuencia de cada clase en entrenamiento
        frecuencias = y.iloc[idx_train].value_counts(normalize=True).reindex(range(len(CLASES))).to_numpy()
        p_base = np.tile(frecuencias, (len(idx_test), 1))

        ll_modelo = log_loss(y.iloc[idx_test], p_test, labels=range(len(CLASES)))
        ll_base = log_loss(y.iloc[idx_test], p_base, labels=range(len(CLASES)))
        filas.append({
            "parte": parte,
            "pitchers_prueba": grupos.iloc[idx_test].nunique(),
            "lanzamientos_prueba": len(idx_test),
            "arboles": modelo.best_iteration_,
            "logloss_modelo": ll_modelo,
            "logloss_baseline": ll_base,
            "mejora_%": 100 * (ll_base - ll_modelo) / ll_base,
        })
        print(f"Parte {parte}/{N_PARTES}: modelo {ll_modelo:.4f} | baseline {ll_base:.4f} | "
              f"mejora {filas[-1]['mejora_%']:.2f}% | {modelo.best_iteration_} árboles")

    return pd.DataFrame(filas), pred_oof


def revisar_calibracion(df, pred_oof):
    """Compara, por clase, la probabilidad promedio predicha con la frecuencia real.

    Si el modelo está bien calibrado, las dos columnas son casi iguales.
    """
    real = df["outcome_id"].value_counts(normalize=True).reindex(range(len(CLASES))).to_numpy()
    tabla = pd.DataFrame({
        "clase": CLASES,
        "frecuencia_real_%": 100 * real,
        "prob_promedio_predicha_%": 100 * pred_oof.mean(axis=0),
    })
    return tabla.round(2)


def main():
    df = agregar_resultado(cargar_datos()).reset_index(drop=True)
    print(f"Lanzamientos: {len(df):,} | pitchers: {df[GRUPO].nunique():,}\n")

    resultados, pred_oof = validacion_cruzada(df)

    print("\nResumen de la validación cruzada (por pitcher):")
    print(resultados.round(4).to_string(index=False))
    print(f"\nMejora promedio sobre el baseline: {resultados['mejora_%'].mean():.2f}% "
          f"(peor parte {resultados['mejora_%'].min():.2f}%, mejor {resultados['mejora_%'].max():.2f}%)")

    print("\nCalibración general:")
    print(revisar_calibracion(df, pred_oof).to_string(index=False))

    print("\nEjemplo de 5 lanzamientos (probabilidades fuera de muestra):")
    ejemplo = pd.concat([df[FEATURES + ["outcome_class"]].iloc[:5].reset_index(drop=True),
                         pd.DataFrame(pred_oof[:5], columns=[f"p_{c}" for c in CLASES]).round(3)], axis=1)
    print(ejemplo.to_string(index=False))

    # ---- guardar predicciones fuera de muestra (para la Tarea 9)
    CARPETA_SALIDAS.mkdir(exist_ok=True)
    columnas_id = [c for c in ["PitchUID", "pitch_id", GRUPO] if c in df.columns]
    salida = df[columnas_id + ["outcome_class"]].copy()
    for i, clase in enumerate(CLASES):
        salida[f"p_{clase}"] = pred_oof[:, i]
    salida.to_parquet(CARPETA_SALIDAS / "predicciones_oof.parquet", index=False)
    resultados.to_csv(CARPETA_SALIDAS / "validacion_multiclase.csv", index=False)

    # ---- modelo final: entrenado con TODOS los datos (para el dashboard y para pitches nuevos)
    modelo_final = entrenar_con_early_stopping(df[FEATURES], df["outcome_id"], df[GRUPO])
    CARPETA_MODELOS.mkdir(exist_ok=True)
    modelo_final.booster_.save_model(str(CARPETA_MODELOS / "modelo_resultado.txt"))

    print(f"\nGuardado: {CARPETA_SALIDAS / 'predicciones_oof.parquet'}")
    print(f"Guardado: {CARPETA_SALIDAS / 'validacion_multiclase.csv'}")
    print(f"Guardado: {CARPETA_MODELOS / 'modelo_resultado.txt'}")


if __name__ == "__main__":
    main()
