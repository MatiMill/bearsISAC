import pandas as pd
from pathlib import Path

RUTA = Path(__file__).resolve().parent.parent/ "stuff_model_df.parquet"

def cargar_datos():
    df = pd.read_parquet(RUTA)
    
    #tipos de tiro reconocidos y pitch calls
    df = df[~df["PitchCall"].isin(["BallIntentional", "AutomaticBall","AutomaticStrike", "Undefined"])]
    tipos_validos = ["Four-Seam", "Slider", "Sinker", "Changeup","Curveball", "Cutter", "Splitter", "Sweeper"]
    df = df[df["AutoPitchType"].isin(tipos_validos)]

    #pitchers con mano reconocida
    df = df[df["PitcherThrows"].isin(["Right", "Left"])]

    #eliminar datos con columnas importantes vacias
    #tenemos que agregar categorias aca cuando implementemos cualquier dato nuevo
    cols_importantes = ["RelSpeed", "SpinRate", "InducedVertBreak",
                        "HorzBreak", "Extension", "RelHeight", "RelSide", "pitcher_anon_id"]
    df = df.dropna(subset=cols_importantes)

    #quita valores imposibles de a pelota
    df = df[df["RelSpeed"].between(60, 105)]
    df = df[df["Extension"].between(3, 8)]
    df = df[df["InducedVertBreak"].abs() <= 30]
    df = df[df["HorzBreak"].abs() <= 30]

    #voltear para zurdos 
    zurdo = df["PitcherThrows"] == "Left"
    df.loc[zurdo, "HorzBreak"] *= -1
    df.loc[zurdo, "RelSide"] *= -1

    return df

#Release speed - Que tan rapido suelta la pelota
#Spinning rate de pelota - Que tan rapido rota la pelota
#InducedVertBreak - Movimiento vertical debido al giro
#HorzBreak - Movimiento Horizontal 
#Extension - Que tan adelante de la "base" se suelta la pelota
#RelHeight - Altura de release
#RelSide - Posicion de la mano al soltar la pelota en relacion al centro de la "base"