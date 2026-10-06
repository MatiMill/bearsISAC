"""
Tarea 6: modelo multiclase de resultado del lanzamiento (Stuff+ v1, SIN altitud).

Para cada lanzamiento predice 5 probabilidades que suman 1:
    P(ball), P(called_strike), P(whiff), P(foul), P(in_play)

Es UN solo modelo (no uno por conteo): la física del pitch más el conteo
(Balls, Strikes) como features (opción B). El conteo entra como feature porque
hay sesgo de selección (los sliders se usan en 0-2, las rectas en 3-0).

Altitud: esta versión NO la usa. Para agregarla (v2) basta con añadir la columna
"densidad_juego" a FEATURES; todo el resto del código sale de esa lista.

Qué hace, en orden:
  1. Carga y limpia los datos, agrega el target (objetivo.py) y se queda SOLO con
     el conjunto de entrenamiento (particion.py); el test sellado no se toca.
  2. Validación cruzada de 5 partes agrupada por pitcher (GroupKFold):
     cada pitcher cae en prueba exactamente una vez y nunca está en entrenamiento
     y prueba a la vez.
  3. En cada parte compara la log-loss del modelo contra un baseline que predice
     la frecuencia de cada clase DENTRO DE CADA CONTEO (calculada en el train del fold).
  4. Revisa la calibración general y por conteo (prob. predicha vs frecuencia real).
  5. Guarda las predicciones fuera de muestra (insumo de la Tarea 9), los 5
     modelos de los folds y un modelo final entrenado con todo el entrenamiento.

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
from particion import separar_test

# ---------------------------------------------------------------- configuración
FISICA = ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak",
          "Extension", "RelHeight", "RelSide",
          "SpinAxis"]                      # dirección del giro (ya volteada para zurdos en limpieza.py)
CONTEO = ["Balls", "Strikes"]              # opción B: el conteo entra como feature
# Para altitud v2: FEATURES = FISICA + CONTEO + ["densidad_juego"]  (y nada más).
FEATURES = FISICA + CONTEO
# VertRelAngle NO se usa en Stuff+: dentro de cada tipo de pitch correlaciona 0.67-0.84 con
# PlateLocHeight, o sea, codifica la ubicación (a dónde apuntó el pitcher), no la calidad.
# Con él la mejora sobre el baseline sube de 1.04% a 12.35%, casi toda por ubicación.
# Sí sirve para Location+ / Pitching+, donde la ubicación está permitida.
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
    min_child_samples=1000,    # elegido por CV (busqueda_min_child.py): menor log-loss;
                               # 500 es indistinguible (pareado) pero se prefiere el más
                               # regularizado, y 2000 pierde en 5/5 folds
    subsample=0.8,             # cada árbol ve el 80% de las filas...
    subsample_freq=1,
    colsample_bytree=0.8,      # ...y el 80% de las columnas
    random_state=SEMILLA,
    verbose=-1,
)


def entrenar_con_early_stopping(X, y, grupos, parametros=PARAMETROS):
    """Entrena un modelo apartando el 10% de los pitchers para decidir cuándo parar.

    Early stopping: después de cada árbol se mide la log-loss en esos pitchers
    apartados. Cuando deja de bajar durante 50 árboles seguidos, se para y se
    usa el número de árboles que dio la mejor log-loss.
    """
    corte = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEMILLA)
    idx_fit, idx_val = next(corte.split(X, y, groups=grupos))

    modelo = lgb.LGBMClassifier(**parametros)
    modelo.fit(
        X.iloc[idx_fit], y.iloc[idx_fit],
        eval_set=[(X.iloc[idx_val], y.iloc[idx_val])],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return modelo


def baseline_por_conteo(y_train, conteo_train, conteo_test):
    """Probabilidades del baseline: frecuencia de cada clase dentro de cada conteo.

    Se calcula con el train del fold. Un conteo que no aparezca en el train usa la
    frecuencia global del train como respaldo.
    """
    n = len(CLASES)
    global_freq = y_train.value_counts(normalize=True).reindex(range(n)).fillna(0).to_numpy()

    tabla = (pd.DataFrame({"y": y_train.to_numpy(),
                           "Balls": conteo_train["Balls"].to_numpy(),
                           "Strikes": conteo_train["Strikes"].to_numpy()})
             .groupby(["Balls", "Strikes"])["y"]
             .value_counts(normalize=True)
             .unstack("y").reindex(columns=range(n)).fillna(0))

    p = (conteo_test.merge(tabla, left_on=["Balls", "Strikes"], right_index=True, how="left")
         [list(range(n))].to_numpy().copy())
    sin_conteo = np.isnan(p).any(axis=1)
    p[sin_conteo] = global_freq
    return p


def validacion_cruzada(df, parametros=PARAMETROS):
    """Devuelve (tabla por parte, predicciones fuera de muestra, 5 modelos, tabla por clase)."""
    n_clases = len(CLASES)
    X, y, grupos = df[FEATURES], df["outcome_id"], df[GRUPO]
    conteo = df[CONTEO]
    pred_oof = np.full((len(df), n_clases), np.nan)
    fold_de_fila = np.full(len(df), -1, dtype=int)   # a qué fold de prueba cae cada fila
    filas, modelos = [], []
    # sumas (no promedios) de log-loss binaria por clase, para agregar sobre todo el OOF
    suma_ll_modelo = np.zeros(n_clases)
    suma_ll_base = np.zeros(n_clases)

    for parte, (idx_train, idx_test) in enumerate(GroupKFold(n_splits=N_PARTES).split(X, y, grupos), start=1):
        modelo = entrenar_con_early_stopping(X.iloc[idx_train], y.iloc[idx_train], grupos.iloc[idx_train], parametros)
        assert list(modelo.classes_) == list(range(n_clases)), "falta alguna clase en el entrenamiento"
        modelos.append(modelo)

        p_test = modelo.predict_proba(X.iloc[idx_test])
        pred_oof[idx_test] = p_test
        fold_de_fila[idx_test] = parte

        # baseline: frecuencia de cada clase dentro de cada conteo (train del fold)
        p_base = baseline_por_conteo(y.iloc[idx_train], conteo.iloc[idx_train], conteo.iloc[idx_test])

        # log-loss binaria por clase (una clase vs. las demás), sumada para agrupar al final
        y_test = y.iloc[idx_test].to_numpy()
        for c in range(n_clases):
            y_bin = (y_test == c).astype(int)
            suma_ll_modelo[c] += log_loss(y_bin, p_test[:, c], labels=[0, 1], normalize=False)
            suma_ll_base[c] += log_loss(y_bin, p_base[:, c], labels=[0, 1], normalize=False)

        ll_modelo = log_loss(y_test, p_test, labels=range(n_clases))
        ll_base = log_loss(y_test, p_base, labels=range(n_clases))
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

    n = len(df)
    por_clase = pd.DataFrame({
        "clase": CLASES,
        "logloss_bin_modelo": suma_ll_modelo / n,
        "logloss_bin_baseline": suma_ll_base / n,
    })
    por_clase["mejora_%"] = 100 * (por_clase["logloss_bin_baseline"] - por_clase["logloss_bin_modelo"]) / por_clase["logloss_bin_baseline"]
    return pd.DataFrame(filas), pred_oof, modelos, por_clase.round(4), fold_de_fila


def revisar_calibracion(df, pred_oof):
    """Por clase: probabilidad promedio predicha vs. frecuencia real (todo el conjunto)."""
    real = df["outcome_id"].value_counts(normalize=True).reindex(range(len(CLASES))).to_numpy()
    tabla = pd.DataFrame({
        "clase": CLASES,
        "frecuencia_real_%": 100 * real,
        "prob_promedio_predicha_%": 100 * pred_oof.mean(axis=0),
    })
    return tabla.round(2)


def revisar_calibracion_por_conteo(df, pred_oof):
    """Por cada uno de los 12 conteos y cada clase: prob. predicha vs frecuencia real."""
    outcome = df["outcome_id"].to_numpy()
    filas = []
    for (b, s), posiciones in df.groupby(CONTEO).groups.items():
        pos = np.asarray(posiciones)
        real = pd.Series(outcome[pos]).value_counts(normalize=True).reindex(range(len(CLASES))).fillna(0).to_numpy()
        pred = pred_oof[pos].mean(axis=0)
        for i, clase in enumerate(CLASES):
            filas.append({
                "Balls": int(b), "Strikes": int(s), "n": len(pos), "clase": clase,
                "frecuencia_real_%": round(100 * real[i], 2),
                "prob_predicha_%": round(100 * pred[i], 2),
                "dif_pp": round(100 * (pred[i] - real[i]), 2),
            })
    return pd.DataFrame(filas)


def main():
    df = agregar_resultado(cargar_datos()).reset_index(drop=True)
    entrenamiento, _test = separar_test(df)
    df = entrenamiento.reset_index(drop=True)          # el test sellado no se toca
    print(f"Entrenamiento: {len(df):,} lanzamientos | {df[GRUPO].nunique():,} pitchers "
          f"(test sellado apartado: {len(_test):,} lanzamientos)\n")

    resultados, pred_oof, modelos, por_clase, fold_de_fila = validacion_cruzada(df)

    print("\nResumen de la validación cruzada (por pitcher):")
    print(resultados.round(4).to_string(index=False))
    print(f"\nMejora promedio sobre el baseline: {resultados['mejora_%'].mean():.2f}% "
          f"(peor parte {resultados['mejora_%'].min():.2f}%, mejor {resultados['mejora_%'].max():.2f}%)")

    print("\nMejora de log-loss binaria por clase (esa clase vs. las demás; modelo vs baseline por conteo):")
    print(por_clase.to_string(index=False))

    print("\nCalibración general:")
    print(revisar_calibracion(df, pred_oof).to_string(index=False))

    print("\nCalibración por conteo (12 conteos × 5 clases):")
    print(revisar_calibracion_por_conteo(df, pred_oof).to_string(index=False))

    # ---- guardar predicciones fuera de muestra (contrato para la Tarea 9)
    CARPETA_SALIDAS.mkdir(exist_ok=True)
    columnas_id = [c for c in ["PitchUID", GRUPO] if c in df.columns]
    salida = df[columnas_id + CONTEO + ["outcome_class"]].copy()
    for i, clase in enumerate(CLASES):
        salida[f"p_{clase}"] = pred_oof[:, i]
    salida.to_parquet(CARPETA_SALIDAS / "predicciones_oof.parquet", index=False)
    resultados.to_csv(CARPETA_SALIDAS / "validacion_multiclase.csv", index=False)

    # ---- asignación de folds (pitcher -> fold de prueba), para que predecir_12_conteos
    # use EXACTAMENTE la misma partición out-of-fold sin reconstruirla
    folds = (pd.DataFrame({GRUPO: df[GRUPO].to_numpy(), "fold": fold_de_fila})
             .drop_duplicates().sort_values(GRUPO))
    folds.to_csv(CARPETA_SALIDAS / "folds.csv", index=False)

    # ---- guardar los 5 modelos de los folds
    CARPETA_MODELOS.mkdir(exist_ok=True)
    for parte, modelo in enumerate(modelos, start=1):
        modelo.booster_.save_model(str(CARPETA_MODELOS / f"modelo_resultado_fold{parte}.txt"),
                                   num_iteration=modelo.best_iteration_)

    # ---- modelo final: entrenado con TODO el entrenamiento (para el dashboard y pitches nuevos)
    modelo_final = entrenar_con_early_stopping(df[FEATURES], df["outcome_id"], df[GRUPO])
    modelo_final.booster_.save_model(str(CARPETA_MODELOS / "modelo_resultado.txt"),
                                     num_iteration=modelo_final.best_iteration_)

    print(f"\nGuardado: {CARPETA_SALIDAS / 'predicciones_oof.parquet'}")
    print(f"Guardado: {CARPETA_SALIDAS / 'validacion_multiclase.csv'}")
    print(f"Guardado: {CARPETA_SALIDAS / 'folds.csv'} ({len(folds):,} pitchers)")
    print(f"Guardado: {CARPETA_MODELOS / 'modelo_resultado_fold{1..5}.txt'} (5 folds)")
    print(f"Guardado: {CARPETA_MODELOS / 'modelo_resultado.txt'} (final)")


if __name__ == "__main__":
    main()
