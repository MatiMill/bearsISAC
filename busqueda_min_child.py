"""
Paso 6: búsqueda de min_child_samples por validación cruzada (por pitcher).

Prueba {20, 50, 100, 200, 500, 1000, 2000} con la MISMA CV GroupKFold por
pitcher sobre el entrenamiento (test 2026 sellado, no entra). No guarda modelos
ni predicciones: solo mide la log-loss de cada valor fold a fold.

Regla de elección (corregida 2026-10-05): la desviación de la log-loss ENTRE
folds está dominada por la distinta dificultad de cada fold, que afecta igual a
todos los valores, así que no sirve para desempatar. Se usa la comparación
PAREADA fold a fold (misma partición → se cancela el término común de dificultad):
  - elegir el de menor log-loss promedio;
  - si la diferencia pareada fold a fold con otro valor MÁS GRANDE no es
    consistente (ese valor menor no gana en al menos 4 de 5 folds), preferir el
    más grande (más regularización → Stuff+ más estable).

Uso:
    python busqueda_min_child.py
"""
import numpy as np
import pandas as pd

from limpieza import cargar_datos
from objetivo import agregar_resultado
from particion import separar_test
from modelo_multiclase import PARAMETROS, GRUPO, validacion_cruzada

VALORES = [20, 50, 100, 200, 500, 1000, 2000]
REFERENCIA = 1000       # valor contra el que se hace la comparación pareada
MIN_FOLDS_CONSISTENTE = 4   # "gana consistentemente" = en al menos 4 de 5 folds


def main():
    df = agregar_resultado(cargar_datos()).reset_index(drop=True)
    entrenamiento, _test = separar_test(df)
    df = entrenamiento.reset_index(drop=True)
    print(f"Entrenamiento: {len(df):,} lanzamientos | {df[GRUPO].nunique():,} pitchers\n")

    ll_por_valor = {}       # min_child_samples -> log-loss de los 5 folds
    for m in VALORES:
        print(f"--- min_child_samples = {m} ---")
        resultados, _oof, _modelos, _clase = validacion_cruzada(df, {**PARAMETROS, "min_child_samples": m})
        ll_por_valor[m] = resultados["logloss_modelo"].to_numpy()
        print()

    ref = ll_por_valor[REFERENCIA]
    filas = []
    for m in VALORES:
        dif = ll_por_valor[m] - ref                     # >0 = m peor que la referencia
        folds_gana_ref = int((ref < ll_por_valor[m]).sum())   # folds donde 1000 es mejor que m
        filas.append({
            "min_child_samples": m,
            "logloss_media": ll_por_valor[m].mean(),
            f"dif_vs_{REFERENCIA}_media": dif.mean(),
            f"dif_vs_{REFERENCIA}_std": dif.std(ddof=1),
            f"folds_{REFERENCIA}_gana": folds_gana_ref,
        })
    tabla = pd.DataFrame(filas)

    # ---- elección con la regla corregida
    v_best = int(tabla.loc[tabla["logloss_media"].idxmin(), "min_child_samples"])
    mayores_indistinguibles = [
        m for m in VALORES
        if m > v_best and int((ll_por_valor[v_best] < ll_por_valor[m]).sum()) < MIN_FOLDS_CONSISTENTE
    ]
    elegido = max(mayores_indistinguibles) if mayores_indistinguibles else v_best

    print("Resultado de la búsqueda (comparación pareada contra 1000):")
    print(tabla.round(5).to_string(index=False))
    print(f"\nMenor log-loss promedio: min_child_samples={v_best}.")
    print(f"Valores más grandes que {v_best} e indistinguibles de él (1000 no gana en >=4/5 folds): "
          f"{mayores_indistinguibles if mayores_indistinguibles else 'ninguno'}.")
    print(f"Elegido: min_child_samples={elegido}.")


if __name__ == "__main__":
    main()
