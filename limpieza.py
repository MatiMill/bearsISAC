import pandas as pd
from pybaseball import statcast

def cargar_datos():
    df = statcast(start_dt="2025-07-01", end_dt="2025-07-07")

    #print("Before:", len(df))
    #Creates mask to identify all non comp pitches and no pitch throwns and removes them 
    non_comp = ["Eephus", "Pitch Out", "Other"]
    df = df[~df["pitch_name"].isin(non_comp)]

    no_pitch_thrown = ["automatic_ball", "automatic_strike", "pitchout"]
    df = df[~df["description"].isin(no_pitch_thrown)]

    #Quita datos vacios de las columnas mas importantes 
    cols_importantes = ["pitch_name", "release_speed", "release_spin_rate", "pfx_x", "pfx_z"]
    #borra pitches con celdas importantes vacias
    df = df.dropna(subset=cols_importantes)

    #print("After:", len(df))

    #Conversion de pies a pulgadas de datos de posicion
    df["pfx_x"] = df["pfx_x"] * 12
    df["pfx_z"] = df["pfx_z"] * 12

    #flip sides for x coord
    df.loc[df["p_throws"] == "L", "pfx_x"] *= -1
    
    return df