"""
Paso 4: separar el test sellado.

La última temporada (year más grande) se aparta como test y no se toca hasta la
validación final. Todo lo demás (hiperparámetros, early stopping, selección de
features) se decide solo con las temporadas de entrenamiento usando GroupKFold
por pitcher. Si se eligieran los hiperparámetros con los mismos datos que se
reportan, el resultado saldría inflado.

Uso:
    python particion.py     # imprime el reporte del split sobre los datos limpios
"""

import os

PITCHER = "pitcher_anon_id"
# Modo final (BEARS_FINAL=1): después de validar en el test sellado con validacion_test.py, la temporada sellada pasa a
# ser parte del entrenamiento para los scores finales. Sin la variable, todo sigue como antes (2026 sellado).
MODO_FINAL = os.environ.get("BEARS_FINAL") == "1"


def separar_test(df):
    """Separa el test sellado de los datos de entrenamiento.

    El test es la última temporada (year más grande); el entrenamiento es todo
    lo demás. Funciona con cualquier df que tenga la columna 'year'.

    Devuelve (entrenamiento, test). En modo final (BEARS_FINAL=1) el entrenamiento es TODO el df y `test` queda
    solo como la última temporada (ya validada), que los scripts no usan salvo para imprimir su tamaño.
    """
    ultima = df["year"].max()
    test = df[df["year"] == ultima]
    entrenamiento = df if MODO_FINAL else df[df["year"] != ultima]
    return entrenamiento, test


def revisar(df, entrenamiento, test):
    """Imprime el tamaño de cada parte y el solapamiento de pitchers."""
    ultima = df["year"].max()
    print(f"Test sellado = temporada {ultima} (la última). No se toca hasta la validación final.")

    print("\n1) Cuántos lanzamientos hay en cada parte")
    for nombre, parte in [("entrenamiento", entrenamiento), ("test", test)]:
        print(f"   {nombre:<13} {len(parte):>8,}  ({100 * len(parte) / len(df):4.1f}%)")
    print(f"   {'total':<13} {len(df):>8,}")

    print("\n2) Cuántos pitchers distintos hay en cada parte")
    pitchers_train = set(entrenamiento[PITCHER].unique())
    pitchers_test = set(test[PITCHER].unique())
    print(f"   entrenamiento {len(pitchers_train):>6,}")
    print(f"   test          {len(pitchers_test):>6,}")

    print("\n3) Pitchers del test: ¿aparecen también en entrenamiento?")
    vistos = pitchers_test & pitchers_train
    nuevos = pitchers_test - pitchers_train
    print(f"   vistos en entrenamiento {len(vistos):>6,}  ({100 * len(vistos) / len(pitchers_test):4.1f}% del test)")
    print(f"   nuevos (solo en test)   {len(nuevos):>6,}  ({100 * len(nuevos) / len(pitchers_test):4.1f}% del test)")


if __name__ == "__main__":
    from limpieza import cargar_datos

    datos = cargar_datos()
    entrenamiento, test = separar_test(datos)
    revisar(datos, entrenamiento, test)
