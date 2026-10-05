"""
Tarea 5: construir el target de 5 resultados.

Cada lanzamiento termina en uno de 5 resultados. Esta es la "respuesta correcta"
que el modelo multiclase (modelo_multiclase.py) aprende a predecir.

    ball           el bateador no hizo swing y fue bola
    called_strike  el bateador no hizo swing y fue strike
    whiff          hizo swing y falló
    foul           hizo swing, tocó la bola y salió de foul
    in_play        hizo swing y metió la bola al campo

Uso:
    python objetivo.py      # imprime las revisiones sobre los datos reales
"""
import pandas as pd

# El orden importa: el modelo numera las clases 0, 1, 2, 3, 4 en este orden,
# y las columnas de probabilidades salen en este mismo orden.
CLASES = ["ball", "called_strike", "whiff", "foul", "in_play"]
ID_CLASE = {clase: i for i, clase in enumerate(CLASES)}

# Traducción de PitchCall (nombres de Trackman) a nuestras 5 clases.
# Si la revisión encuentra un PitchCall que no está aquí, hay que decidir
# a qué clase va, agregarlo y anotar la decisión en docs/decisions.md.
MAPA_PITCHCALL = {
    "BallCalled": "ball",
    "BallinDirt": "ball",
    "HitByPitch": "ball",  # decisión v0: el pelotazo cuenta como bola (el bateador no hizo swing y no fue strike)
    "StrikeCalled": "called_strike",
    "StrikeSwinging": "whiff",
    "FoulBall": "foul",
    "FoulBallNotFieldable": "foul",
    "FoulBallFieldable": "foul",
    "InPlay": "in_play",
}

# Qué clases son swing. Sirve para comprobar que coincidimos con la columna is_swing del ISAC.
CLASES_SWING = {"whiff", "foul", "in_play"}


def agregar_resultado(df):
    """Agrega las columnas outcome_class (texto) y outcome_id (0 a 4).

    Las filas cuyo PitchCall no está en MAPA_PITCHCALL se quitan, y se avisa cuáles fueron.
    """
    df = df.copy()
    df["outcome_class"] = df["PitchCall"].map(MAPA_PITCHCALL)

    sin_clase = df.loc[df["outcome_class"].isna(), "PitchCall"].value_counts(dropna=False)
    if len(sin_clase) > 0:
        print("AVISO: estos PitchCall no tienen clase y se van a quitar:")
        for valor, n in sin_clase.items():
            print(f"   {valor}: {n:,} filas")
        df = df.dropna(subset=["outcome_class"])

    df["outcome_id"] = df["outcome_class"].map(ID_CLASE).astype(int)
    return df


def revisar(df):
    """Imprime las tres revisiones de la Tarea 5."""
    print("\n1) Cuántos lanzamientos hay de cada clase")
    conteo = df["outcome_class"].value_counts().reindex(CLASES)
    tabla = pd.DataFrame({"lanzamientos": conteo, "%": (100 * conteo / len(df)).round(1)})
    print(tabla.to_string())

    if "is_whiff" in df.columns:
        print("\n2) ¿Coincide nuestra clase 'whiff' con la columna is_whiff del ISAC?")
        cruce = pd.crosstab(df["outcome_class"] == "whiff", df["is_whiff"],
                            rownames=["es whiff (nuestro)"], colnames=["is_whiff (ISAC)"])
        print(cruce.to_string())
        fuera_de_diagonal = ((df["outcome_class"] == "whiff") != (df["is_whiff"] == 1)).sum()
        print("   OK: coinciden en todas las filas" if fuera_de_diagonal == 0
              else f"   OJO: {fuera_de_diagonal:,} filas no coinciden. Revisar qué significa is_whiff.")

    if "is_swing" in df.columns:
        print("\n3) ¿Coincide nuestra idea de swing con la columna is_swing del ISAC?")
        nuestro_swing = df["outcome_class"].isin(CLASES_SWING)
        cruce = pd.crosstab(nuestro_swing, df["is_swing"],
                            rownames=["es swing (nuestro)"], colnames=["is_swing (ISAC)"])
        print(cruce.to_string())
        fuera_de_diagonal = (nuestro_swing != (df["is_swing"] == 1)).sum()
        print("   OK: coinciden en todas las filas" if fuera_de_diagonal == 0
              else f"   OJO: {fuera_de_diagonal:,} filas no coinciden. Ver cuáles PitchCall son con:\n"
                   "   df[nuestro_swing != (df.is_swing == 1)].PitchCall.value_counts()")


if __name__ == "__main__":
    from limpieza import cargar_datos

    datos = agregar_resultado(cargar_datos())
    revisar(datos)
