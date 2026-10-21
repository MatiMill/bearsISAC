"""
Altitud, Idea 4: features de arsenal (relacionales) para Stuff+.

Un pitch se califica también por cómo se compara con la recta del mismo pitcher: un cambio 10 mph más lento que
su recta no es lo mismo que uno 5 mph más lento. Por pitcher × temporada:

    recta primaria   el tipo más usado entre Four-Seam, Sinker y Cutter
    velo_diff_vs_fb  RelSpeed del pitch − mediana de RelSpeed de su recta primaria           (mph)
    ivb_diff_vs_fb   IVB_n   del pitch − mediana de IVB_n   de su recta primaria              (pulgadas)
    hb_diff_vs_fb    HB_n    del pitch − mediana de HB_n    de su recta primaria              (pulgadas)
    release_diff_vs_fb  RelHeight del pitch − mediana de RelHeight de su recta primaria       (pies)
    pitch_usage_pct  % de los pitches del pitcher en esa temporada que son de este tipo

Altitud: IVB_n y HB_n son el movimiento llevado a nivel del mar (altitud_movimiento.py), así que las diferencias
con la recta no cambian por el aire del juego en que se midieron. Al evaluar un pitch en OTRO aire, la recta
también cambia de movimiento (ver `diferencias_en_aire`).

Los pitchers sin recta (Four-Seam/Sinker/Cutter) en esa temporada quedan con NaN (LightGBM los maneja).
Solo se usan las propias mediciones del pitcher y temporada: no entra ningún resultado.

Uso desde otros scripts:
    from arsenal import agregar_arsenal
"""
import numpy as np
import pandas as pd

from altitud_movimiento import RHO_MAR, ajustar_movimiento
from modelo_multiclase import GRUPO

RECTAS = ["Four-Seam", "Sinker", "Cutter"]
COLS_ARSENAL = ["velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb", "release_diff_vs_fb", "pitch_usage_pct"]


def agregar_arsenal(df):
    """Agrega fb_tipo y las 5 columnas de COLS_ARSENAL. Requiere IVB_n, HB_n (altitud_movimiento.normalizar_movimiento)."""
    d = df.copy()
    d["_tipo"] = d["AutoPitchType"].astype(str)
    llave = [GRUPO, "year"]

    n_tipo = d.groupby(llave + ["_tipo"]).size().rename("n_tipo").reset_index()
    n_total = d.groupby(llave).size().rename("n_total").reset_index()
    # recta primaria: la más usada entre las rectas (en empate, la de nombre menor: resultado determinista)
    rectas = (n_tipo[n_tipo["_tipo"].isin(RECTAS)].sort_values(llave + ["n_tipo", "_tipo"], ascending=[True, True, False, True])
              .drop_duplicates(llave)[llave + ["_tipo"]].rename(columns={"_tipo": "fb_tipo"}))
    med = (d.groupby(llave + ["_tipo"]).agg(fb_v=("RelSpeed", "median"), fb_i=("IVB_n", "median"),
                                             fb_h=("HB_n", "median"), fb_r=("RelHeight", "median")).reset_index())
    fb = rectas.merge(med, left_on=llave + ["fb_tipo"], right_on=llave + ["_tipo"]).drop(columns="_tipo")

    d = (d.merge(fb, on=llave, how="left").merge(n_tipo, on=llave + ["_tipo"], how="left")
         .merge(n_total, on=llave, how="left"))
    d["velo_diff_vs_fb"] = d["RelSpeed"] - d["fb_v"]
    d["ivb_diff_vs_fb"] = d["IVB_n"] - d["fb_i"]
    d["hb_diff_vs_fb"] = d["HB_n"] - d["fb_h"]
    d["release_diff_vs_fb"] = d["RelHeight"] - d["fb_r"]
    d["pitch_usage_pct"] = 100 * d["n_tipo"] / d["n_total"]
    assert len(d) == len(df), "el merge de arsenal cambió el número de filas"
    return d.drop(columns=["_tipo", "fb_v", "fb_i", "fb_h", "fb_r", "n_tipo", "n_total"]).set_index(df.index)


def diferencias_en_aire(df, rho_destino):
    """ivb_diff_vs_fb y hb_diff_vs_fb de cada pitch si él Y su recta se vieran con `rho_destino`.

    Parte del movimiento a nivel del mar del pitch (IVB_n, HB_n) y del de su recta (el del pitch menos la diferencia
    guardada) y lleva cada uno a `rho_destino` con SU tipo (el ajuste ya no es aditivo: depende de cuánto se mueve).
    La velocidad, el release y el uso no cambian con el aire.
    """
    tipo_p = df["AutoPitchType"].astype(str)
    tipo_r = df["fb_tipo"].fillna("TODOS").astype(str)
    ivb_p, hb_p = ajustar_movimiento(df["IVB_n"].to_numpy(), df["HB_n"].to_numpy(), RHO_MAR, rho_destino, tipo=tipo_p)
    fb_ivb_mar = df["IVB_n"].to_numpy() - df["ivb_diff_vs_fb"].to_numpy()
    fb_hb_mar = df["HB_n"].to_numpy() - df["hb_diff_vs_fb"].to_numpy()
    ivb_r, hb_r = ajustar_movimiento(fb_ivb_mar, fb_hb_mar, RHO_MAR, rho_destino, tipo=tipo_r)
    return ivb_p - ivb_r, hb_p - hb_r
