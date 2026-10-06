"""
limpieza.py
Carga los datos del ISAC, quita los pitches que no sirven y voltea a los zurdos.

Uso:
    python limpieza.py    # limpia, imprime la tabla de filas quitadas por filtro
                          # y guarda ../pitches_clean.parquet

Desde otros scripts:
    from limpieza import cargar_datos
    df = cargar_datos()
"""
import pandas as pd
from pathlib import Path

RUTA = Path(__file__).resolve().parent.parent / "stuff_model_df.parquet"
# Fuera del repo, junto al parquet original: también son datos del ISAC
RUTA_LIMPIO = RUTA.parent / "pitches_clean.parquet"

TIPOS_VALIDOS = ["Four-Seam", "Slider", "Sinker", "Changeup",
                 "Curveball", "Cutter", "Splitter", "Sweeper"]

# Si una de estas columnas está vacía, se quita la fila.
# Agregar aquí cualquier columna nueva que el modelo use como feature.
COLS_IMPORTANTES = ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak",
                    "Extension", "RelHeight", "RelSide", "SpinAxis",
                    "VertRelAngle", "pitcher_anon_id"]

# Columnas horizontales que cambian de signo según la mano del pitcher
COLS_VOLTEAR = ["HorzBreak", "RelSide", "HorzRelAngle", "HorzApprAngle"]


def filtrar(df, mascara, nombre, registro):
    """Conserva las filas donde mascara es True y anota cuántas se quitaron."""
    registro.append((nombre, int((~mascara).sum())))
    return df[mascara]


def cargar_datos(mostrar_tabla=False):
    df = pd.read_parquet(RUTA)
    filas_iniciales = len(df)
    registro = []

    # Quitar pitches que no son competencia real
    no_competitivos = ["BallIntentional", "AutomaticBall", "AutomaticStrike", "Undefined"]
    df = filtrar(df, ~df["PitchCall"].isin(no_competitivos), "PitchCall no competitivo", registro)
    # Bunts puestos en juego (los de foul o fallados no vienen marcados)
    df = filtrar(df, df["hit_type"] != "Bunt", "Bunts", registro)

    # Quitar lo que no se puede describir
    df = filtrar(df, df["AutoPitchType"].isin(TIPOS_VALIDOS), "Tipo de pitch no válido", registro)
    df = filtrar(df, df["PitcherThrows"].isin(["Right", "Left"]), "Mano desconocida", registro)
    df = filtrar(df, df[COLS_IMPORTANTES].notna().all(axis=1), "Columnas importantes vacías", registro)

    # Quitar valores físicamente imposibles (errores del radar)
    df = filtrar(df, df["RelSpeed"].between(60, 105), "RelSpeed fuera de 60-105 mph", registro)
    df = filtrar(df, df["Extension"].between(3, 8), "Extension fuera de 3-8 ft", registro)
    df = filtrar(df, df["InducedVertBreak"].abs() <= 30, "|InducedVertBreak| > 30 in", registro)
    df = filtrar(df, df["HorzBreak"].abs() <= 30, "|HorzBreak| > 30 in", registro)
    df = filtrar(df, df["SpinAxis"].between(0, 360), "SpinAxis fuera de 0-360°", registro)

    # Voltear zurdos: después de esto, HorzBreak negativo = hacia el guante, sin importar la mano
    df = df.copy()
    zurdo = df["PitcherThrows"] == "Left"
    for col in COLS_VOLTEAR:
        df.loc[zurdo, col] *= -1
    # SpinAxis es un ángulo (0-360°): el espejo de un ángulo es 360 - ángulo
    df.loc[zurdo, "SpinAxis"] = 360 - df.loc[zurdo, "SpinAxis"]

    if mostrar_tabla:
        tabla = pd.DataFrame(registro, columns=["filtro", "filas_quitadas"])
        tabla["%_del_total"] = (100 * tabla["filas_quitadas"] / filas_iniciales).round(2)
        print(tabla.to_string(index=False))
        print(f"\nFilas iniciales: {filas_iniciales:,}")
        print(f"Filas quitadas:  {filas_iniciales - len(df):,}")
        print(f"Filas finales:   {len(df):,}")

    return df


if __name__ == "__main__":
    datos = cargar_datos(mostrar_tabla=True)
    datos.to_parquet(RUTA_LIMPIO, index=False)
    print("\nGuardado en", RUTA_LIMPIO)
