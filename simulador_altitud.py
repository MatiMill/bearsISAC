"""
Simulador de altitud: califica UN pitch en el aire de nivel del mar y en el de Harp Helú (lo usa dashboard.py).

El coach mete la física de un pitch tal como la midió y el aire en que la midió (`rho_medido`, 1.00 = nivel del mar,
0.830 = Harp Helú). Se hace lo mismo que contrafactual_altitud.py con los pitches de entrenamiento:

    1. el movimiento (y el de su recta de referencia) se lleva a nivel del mar → IVB_n, HB_n y las diferencias
       contra la recta (arsenal.py);
    2. se evalúa en los 12 conteos con Stuff+ v3 en dos escenarios: nivel del mar (movimiento normalizado,
       densidad 1.00) y Harp Helú (el MISMO movimiento normalizado, densidad 0.830: el modelo ya aprendió el costo
       del aire de la densidad; bajar además el movimiento contaría la pérdida dos veces) → xRV neutral al conteo +
       precio de la bola en juego;
    3. xRV → Stuff+ = 100 + 10 z con la media y la sd de la calibración (calibracion_stuff_altitud.json).

El canal de movimiento se calcula aparte (movimiento bajado a Harp Helú, densidad 1.00). El modelo es el PROMEDIO de los
5 modelos de fold (la calibración viene de predicciones out-of-fold; un solo modelo final agrega ruido de ±2.6 puntos
por pitch, tanto como la diferencia real entre pitchers).

Devuelve el Stuff+ en los dos aires y el delta separado en canales: movimiento, resultado y precio.
`stuff_harp_centrado` es el Stuff+ de Harp Helú en su propia escala (100 = pitch promedio de Harp Helú).

Mano: el modelo usa la convención volteada de limpieza.py (HB negativo = hacia el guante, igual para zurdos y
derechos). Si el radar midió a un zurdo sin voltear, pasa `mano="L"` y se voltean HB, RelSide y SpinAxis.

Uso:
    from simulador_altitud import cargar_recursos, calificar_pitch
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from altitud_movimiento import RHO_HARP, RHO_MAR, ajustar_movimiento
from arsenal import RECTAS, diferencias_en_aire
from contrafactual_altitud import RUTA_CALIBRACION, desplazamiento_precio, xrv_neutral
from modelo_multiclase import CARPETA_MODELOS, N_PARTES
from modelo_stuff_arsenal import FEATURES_V3
from scoring import cargar_rv

TIPOS = ["Four-Seam", "Sinker", "Cutter", "Slider", "Sweeper", "Curveball", "Changeup", "Splitter"]
CAMPOS_FISICA = ["RelSpeed", "SpinRate", "Extension", "RelHeight", "RelSide", "SpinAxis"]


class PromedioFolds:
    """Promedia las probabilidades de los modelos de fold (misma interfaz `predict` que un Booster)."""

    def __init__(self, modelos):
        self.modelos = modelos

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.modelos], axis=0)


def cargar_recursos(prefijo=CARPETA_MODELOS / "modelo_stuff_arsenal"):
    """Modelo Stuff+ v3 (promedio de los folds), precios y pesos, y la calibración. Devuelve un dict para calificar_pitch."""
    rv_wide, pesos = cargar_rv()
    folds = [lgb.Booster(model_file=f"{prefijo}_fold{k}.txt") for k in range(1, N_PARTES + 1)]
    return {"modelo": PromedioFolds(folds), "rv_wide": rv_wide, "pesos": pesos,
            "calib": json.loads(RUTA_CALIBRACION.read_text(encoding="utf-8"))}


def voltear_zurdo(fisica, hb, fb=None):
    """Convierte las mediciones de un zurdo sin voltear a la convención del modelo (como limpieza.py)."""
    f = {**fisica, "RelSide": -fisica["RelSide"], "SpinAxis": 360 - fisica["SpinAxis"]}
    fb = None if fb is None else {**fb, "HB": -fb["HB"]}
    return f, -hb, fb


def calificar_pitch(recursos, tipo, fisica, ivb, hb, rho_medido=RHO_MAR, fb=None, uso_pct=33.0, calib=None, mano="R"):
    """Stuff+ de un pitch a nivel del mar y en Harp Helú.

    `fisica`: dict con RelSpeed, SpinRate, Extension, RelHeight, RelSide, SpinAxis. `ivb`, `hb`: movimiento medido
    con densidad `rho_medido`. `fb`: recta primaria del pitcher medida en el mismo aire, dict con tipo, RelSpeed,
    IVB, HB, RelHeight; None solo si este pitch ES su recta primaria (Four-Seam, Sinker o Cutter). `uso_pct`: % de sus
    pitches que son de este tipo. `calib`: dict con mu y sd (por defecto la calibración global). `mano`: "R" o "L"
    (un zurdo con mediciones sin voltear se voltea antes de calificar).
    """
    c = recursos["calib"] if calib is None else calib
    delta_ip, rho_medio = recursos["calib"]["delta_in_play"], recursos["calib"]["rho_medio"]
    if mano == "L":
        fisica, hb, fb = voltear_zurdo(fisica, hb, fb)
    if fb is None:
        if tipo not in RECTAS:
            raise ValueError(f"{tipo} no es una recta: pasa `fb` con la recta primaria del pitcher (en el entrenamiento "
                             "la recta de referencia siempre es Four-Seam, Sinker o Cutter)")
        fb = {"tipo": tipo, "RelSpeed": fisica["RelSpeed"], "IVB": ivb, "HB": hb, "RelHeight": fisica["RelHeight"]}

    ivb_n, hb_n = ajustar_movimiento(ivb, hb, rho_medido, RHO_MAR, tipo=tipo)
    fb_ivb_n, fb_hb_n = ajustar_movimiento(fb["IVB"], fb["HB"], rho_medido, RHO_MAR, tipo=fb["tipo"])
    fila = pd.DataFrame([{**{k: float(fisica[k]) for k in CAMPOS_FISICA}, "IVB_n": float(ivb_n), "HB_n": float(hb_n),
                          "AutoPitchType": tipo, "fb_tipo": fb["tipo"],
                          "velo_diff_vs_fb": float(fisica["RelSpeed"] - fb["RelSpeed"]),
                          "ivb_diff_vs_fb": float(ivb_n - fb_ivb_n), "hb_diff_vs_fb": float(hb_n - fb_hb_n),
                          "release_diff_vs_fb": float(fisica["RelHeight"] - fb["RelHeight"]),
                          "pitch_usage_pct": float(uso_pct)}])

    def escenario(rho_mov, rho_feature):
        base = fila[CAMPOS_FISICA].copy()
        if rho_mov == RHO_MAR:
            base["IVB_n"], base["HB_n"] = fila["IVB_n"], fila["HB_n"]
            base["ivb_diff_vs_fb"], base["hb_diff_vs_fb"] = fila["ivb_diff_vs_fb"], fila["hb_diff_vs_fb"]
        else:
            base["IVB_n"], base["HB_n"] = ajustar_movimiento(fila["IVB_n"], fila["HB_n"], RHO_MAR, rho_mov, tipo=fila["AutoPitchType"])
            base["ivb_diff_vs_fb"], base["hb_diff_vs_fb"] = diferencias_en_aire(fila, rho_mov)
        base[["velo_diff_vs_fb", "release_diff_vs_fb", "pitch_usage_pct"]] = fila[["velo_diff_vs_fb", "release_diff_vs_fb", "pitch_usage_pct"]]
        xrv, ip = xrv_neutral(recursos["modelo"].predict, base, FEATURES_V3, recursos["rv_wide"], recursos["pesos"], rho=rho_feature)
        return float(xrv[0]), float(ip[0]), float(base["IVB_n"].iloc[0]), float(base["HB_n"].iloc[0])

    x_mar, ip_mar, ivb_mar, hb_mar = escenario(RHO_MAR, RHO_MAR)
    x_harp_mov, _, ivb_harp, hb_harp = escenario(RHO_HARP, RHO_MAR)     # solo canal movimiento (y movimiento que se ve en Harp Helú)
    x_harp, ip_harp, _, _ = escenario(RHO_MAR, RHO_HARP)                # escenario Harp Helú: movimiento normalizado + densidad
    precio_mar = desplazamiento_precio(ip_mar, RHO_MAR, delta_ip, rho_medio)
    precio_harp = desplazamiento_precio(ip_harp, RHO_HARP, delta_ip, rho_medio)

    def puntos(dif_xrv):
        return -10 * dif_xrv / c["sd"]

    stuff_mar = 100 + 10 * (-(x_mar + precio_mar) - c["mu"]) / c["sd"]
    stuff_harp = 100 + 10 * (-(x_harp + precio_harp) - c["mu"]) / c["sd"]
    return {"stuff_mar": stuff_mar, "stuff_harp": stuff_harp, "delta": stuff_harp - stuff_mar,
            "stuff_harp_centrado": stuff_harp - recursos["calib"]["delta_total_medio"],
            "delta_movimiento": puntos(x_harp_mov - x_mar), "delta_resultado": puntos(x_harp - x_harp_mov),
            "delta_precio": puntos(precio_harp - precio_mar),
            "ivb_mar": ivb_mar, "hb_mar": hb_mar, "ivb_harp": ivb_harp, "hb_harp": hb_harp}
