"""
Altitud, Idea 2: ajustar el movimiento de un pitch de un aire a otro.

Usa el efecto MEDIDO en la LMB (efecto_altitud.py), no la física teórica: dentro de cada pitcher × tipo,
el movimiento cambia gamma unidades (pulgadas) por cada 1.00 de densidad:

    x_destino = x_origen + gamma_tipo * (rho_destino - rho_origen)       x = IVB, HB

Se usa la forma ADITIVA con el gamma de cada tipo (y no una proporción) porque el HB de varios tipos
promedia cerca de 0 (cutter), donde una proporción no está definida. Límite: no distingue entre dos
pitches del mismo tipo con distinto tamaño de movimiento (el cambio real es algo mayor en los que
se mueven más).

Medido en la LMB, el movimiento cae MÁS que proporcionalmente a la densidad (elasticidad ≈ 1.3, no 1.0 de la
física simple), así que este ajuste es mayor que el de IVB_destino = IVB_origen · rho_destino / rho_origen.

Contrato (guía §6, módulo 10): ajustar_movimiento(ivb, hb, rho_origen, rho_destino) con `tipo` opcional.
Sin `tipo` usa el efecto promedio de todos los pitches.

Uso desde otros scripts:
    from altitud_movimiento import ajustar_movimiento, normalizar_movimiento
"""
import numpy as np
import pandas as pd

from modelo_multiclase import CARPETA_SALIDAS

RHO_MAR = 1.00        # densidad de referencia: 1.00 = mediana de los juegos "No Altitude"
RHO_HARP = 0.828      # densidad media de los juegos en Harp Helú (estadios.parquet)
RUTA_EFECTO = CARPETA_SALIDAS / "efecto_altitud.csv"
# Tipos con pocos pitchers (Splitter: 61, IC de IVB que cruza 0) toman el efecto del tipo más parecido
SUSTITUTO = {"Splitter": "Changeup"}
_CACHE = {}


def cargar_gamma(ruta=RUTA_EFECTO):
    """Dict {(tipo, 'InducedVertBreak'|'HorzBreak'): gamma}, con tipo 'TODOS' = efecto promedio."""
    if ruta not in _CACHE:
        t = pd.read_csv(ruta)
        t = t[t["variable"].isin(["InducedVertBreak", "HorzBreak"])]
        g = {(r.tipo, r.variable): float(r.gamma) for r in t.itertuples()}
        for destino, fuente in SUSTITUTO.items():
            for var in ("InducedVertBreak", "HorzBreak"):
                g[(destino, var)] = g[(fuente, var)]
        _CACHE[ruta] = g
    return _CACHE[ruta]


def _gamma_de(tipo, var, g):
    """gamma(s) para `tipo` (str o serie/array de str); tipos desconocidos usan 'TODOS'."""
    if isinstance(tipo, str) or tipo is None:
        return g.get((tipo, var), g[("TODOS", var)])
    valores = pd.Series(np.asarray(tipo, dtype=object)).map(lambda t: g.get((t, var), g[("TODOS", var)]))
    return valores.to_numpy(dtype=float)


def ajustar_movimiento(ivb, hb, rho_origen, rho_destino, tipo=None, gamma=None):
    """IVB y HB (pulgadas) de un pitch medido con densidad `rho_origen`, vistos con `rho_destino`.

    `tipo`: 'Four-Seam', 'Slider', ... (un valor o uno por pitch). `gamma`: dict de cargar_gamma().
    Devuelve (ivb_destino, hb_destino) con la misma forma que las entradas.
    """
    g = cargar_gamma() if gamma is None else gamma
    dr = np.asarray(rho_destino, dtype=float) - np.asarray(rho_origen, dtype=float)
    ivb_d = np.asarray(ivb, dtype=float) + _gamma_de(tipo, "InducedVertBreak", g) * dr
    hb_d = np.asarray(hb, dtype=float) + _gamma_de(tipo, "HorzBreak", g) * dr
    return ivb_d, hb_d


def normalizar_movimiento(df, rho_ref=RHO_MAR):
    """Agrega IVB_n y HB_n: el movimiento de cada pitch llevado al aire `rho_ref` (por defecto, nivel del mar).

    Requiere las columnas InducedVertBreak, HorzBreak, AutoPitchType y densidad_juego
    (ver efecto_altitud.agregar_densidad).
    """
    out = df.copy()
    out["IVB_n"], out["HB_n"] = ajustar_movimiento(
        out["InducedVertBreak"], out["HorzBreak"], out["densidad_juego"], rho_ref, tipo=out["AutoPitchType"].astype(str))
    return out
