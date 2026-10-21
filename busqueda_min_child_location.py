"""
Búsqueda de min_child_samples para Location+ (misma regla que busqueda_min_child.py).

Prueba {20, 50, 100, 200, 500, 1000, 2000} con la validación de Location+: features de
modelo_location.py y los MISMOS folds de Stuff+ (../salidas/folds.csv). Test 2026 sellado.
No guarda modelos ni predicciones: solo mide la log-loss de cada valor fold a fold.

Regla (la de Stuff+, ver decisions.md Paso 6): elegir el de menor log-loss promedio; si un
valor MÁS GRANDE es indistinguible en la comparación pareada (el menor no le gana en al
menos 4 de 5 folds), preferir el más grande (más regularización).

Resultado (2026-10-06): menor log-loss 1000, pero 2000 indistinguible (1000 gana 3/5) →
elegido 2000 (PARAMETROS_LOCATION en modelo_location.py). Chequeo extra del borde de la
malla: 5000 pierde contra 2000 en 5/5 folds.

Uso:
    python busqueda_min_child_location.py
"""
import pandas as pd

from busqueda_min_child import MIN_FOLDS_CONSISTENTE, REFERENCIA, VALORES
from limpieza import cargar_datos
from modelo_location import FEATURES_LOCATION, leer_folds
from modelo_multiclase import GRUPO, PARAMETROS, validacion_cruzada
from objetivo import agregar_resultado
from particion import separar_test
from zona import preparar_ubicacion


def main():
    df = preparar_ubicacion(agregar_resultado(cargar_datos()))
    df = separar_test(df)[0].reset_index(drop=True)
    print(f"Entrenamiento: {len(df):,} lanzamientos | {df[GRUPO].nunique():,} pitchers\n")
    fold_de_pitcher = leer_folds()

    ll_por_valor = {}       # min_child_samples -> log-loss de los 5 folds
    for m in VALORES:
        print(f"--- min_child_samples = {m} ---")
        resultados, *_ = validacion_cruzada(df, {**PARAMETROS, "min_child_samples": m},
                                            features=FEATURES_LOCATION, fold_de_pitcher=fold_de_pitcher)
        ll_por_valor[m] = resultados["logloss_modelo"].to_numpy()
        print()

    ref = ll_por_valor[REFERENCIA]
    tabla = pd.DataFrame([{
        "min_child_samples": m,
        "logloss_media": ll_por_valor[m].mean(),
        f"dif_vs_{REFERENCIA}_media": (ll_por_valor[m] - ref).mean(),
        f"dif_vs_{REFERENCIA}_std": (ll_por_valor[m] - ref).std(ddof=1),
        f"folds_{REFERENCIA}_gana": int((ref < ll_por_valor[m]).sum()),
    } for m in VALORES])

    v_best = int(tabla.loc[tabla["logloss_media"].idxmin(), "min_child_samples"])
    # Se sube desde el mejor y se PARA en el primer valor claramente peor (v_best pierde ya no en < 4/5 folds sino en >= 4/5):
    # sin el alto, un valor muy grande "indistinguible" podía ganar aunque uno intermedio fuera claramente peor.
    mayores_indistinguibles = []
    for m in sorted(v for v in VALORES if v > v_best):
        if int((ll_por_valor[v_best] < ll_por_valor[m]).sum()) >= MIN_FOLDS_CONSISTENTE:
            break
        mayores_indistinguibles.append(m)
    elegido = max(mayores_indistinguibles) if mayores_indistinguibles else v_best
    folds_best_gana = {m: int((ll_por_valor[v_best] < ll_por_valor[m]).sum()) for m in VALORES if m != v_best}

    print(f"Resultado de la búsqueda Location+ (comparación pareada contra {REFERENCIA}):")
    print(tabla.round(5).to_string(index=False))
    print(f"\nMenor log-loss promedio: min_child_samples={v_best}.")
    print(f"Folds (de 5) en que {v_best} gana a cada otro valor: {folds_best_gana}")
    print(f"Valores más grandes que {v_best} e indistinguibles de él (gana en <{MIN_FOLDS_CONSISTENTE}/5 folds): "
          f"{mayores_indistinguibles if mayores_indistinguibles else 'ninguno'}.")
    print(f"Elegido: min_child_samples={elegido}.")


if __name__ == "__main__":
    main()
