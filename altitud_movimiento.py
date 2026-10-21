"""
Altitud, Idea 2: ajustar el movimiento de un pitch de un aire a otro.

Usa el efecto MEDIDO en la LMB (efecto_altitud.py), no la física teórica. Dentro de cada pitcher × tipo, el cambio
del movimiento por cada 1.00 de densidad crece con cuánto se mueve el pitch:

    dx/drho = g0 + kappa·x          (x = IVB, HB; g0 y kappa por tipo y por variable)

Es el mismo ajuste de la elasticidad medida (≈ 1.3, no 1.0 de la física simple), pero sin dividir entre el movimiento
medio: el HB de varios tipos promedia cerca de 0 (cutter), donde una proporción pura no está definida. Resolviendo la
ecuación (con y = x + g0/kappa, dy/drho = kappa·y):

    x_destino = (x_origen + g0/kappa)·exp(kappa·(rho_destino − rho_origen)) − g0/kappa

Es exactamente invertible y componible (mar → Harp → mar devuelve el pitch original; origen → destino da lo mismo que
origen → mar → destino), que es lo que necesitan la normalización (IVB_n, HB_n) y el simulador. Con kappa ≈ 0 se
reduce al ajuste aditivo x + g0·Δrho. Con un solo gamma por tipo (versión anterior) los pitches de signo contrario al
del tipo (22 % de los cutters en HB, 11 % de sliders en IVB) se movían en sentido contrario al físico.

Contrato (guía §6, módulo 10): ajustar_movimiento(ivb, hb, rho_origen, rho_destino) con `tipo` opcional.
Sin `tipo` usa el efecto promedio de todos los pitches.

Uso desde otros scripts:
    from altitud_movimiento import ajustar_movimiento, normalizar_movimiento
"""
import numpy as np
import pandas as pd

from modelo_multiclase import CARPETA_SALIDAS

RHO_MAR = 1.00        # densidad de referencia: 1.00 = mediana de los juegos "No Altitude"
RHO_HARP = 0.830      # densidad media de los juegos en Harp Helú, temporadas de entrenamiento (estadios.parquet: 0.8298)
RUTA_EFECTO = CARPETA_SALIDAS / "efecto_altitud.csv"
# Tipos con pocos pitchers toman el efecto del tipo más parecido (Splitter: 61 pitchers, IC de IVB que cruza 0;
# Sweeper: < 30 pitchers, sin fila propia: antes caía en el promedio de la liga, que tiene el signo contrario en HB)
SUSTITUTO = {"Splitter": "Changeup", "Sweeper": "Slider"}
_CACHE = {}


def cargar_gamma(ruta=RUTA_EFECTO):
    """Dict {(tipo, 'InducedVertBreak'|'HorzBreak'): (g0, kappa)}, con tipo 'TODOS' = efecto promedio."""
    if ruta not in _CACHE:
        t = pd.read_csv(ruta)
        assert {"g0", "kappa"} <= set(t.columns), "efecto_altitud.csv es de la versión anterior: corre efecto_altitud.py"
        t = t[t["variable"].isin(["InducedVertBreak", "HorzBreak"])]
        g = {(r.tipo, r.variable): (float(r.g0), float(r.kappa)) for r in t.itertuples()}
        for destino, fuente in SUSTITUTO.items():
            for var in ("InducedVertBreak", "HorzBreak"):
                g[(destino, var)] = g[(fuente, var)]
        _CACHE[ruta] = g
    return _CACHE[ruta]


def _coef_de(tipo, var, g):
    """(g0, kappa) para `tipo` (str o serie/array de str); tipos desconocidos usan 'TODOS'."""
    if isinstance(tipo, str) or tipo is None:
        return g.get((tipo, var), g[("TODOS", var)])
    pares = pd.Series(np.asarray(tipo, dtype=object)).map(lambda t: g.get((t, var), g[("TODOS", var)]))
    return (np.array([p[0] for p in pares], dtype=float), np.array([p[1] for p in pares], dtype=float))


def _mover(x, dr, g0, kappa):
    """x_destino = (x + g0/kappa)·exp(kappa·dr) − g0/kappa; con kappa ≈ 0 es x + g0·dr."""
    x = np.asarray(x, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    chico = np.abs(kappa) < 1e-9
    k = np.where(chico, 1.0, kappa)                       # evita dividir entre 0; se descarta abajo
    exacto = (x + g0 / k) * np.exp(k * dr) - g0 / k
    return np.where(chico, x + g0 * dr, exacto)[()]       # [()]: una entrada escalar devuelve un escalar, no un arreglo de 0 dimensiones


def ajustar_movimiento(ivb, hb, rho_origen, rho_destino, tipo=None, gamma=None):
    """IVB y HB (pulgadas) de un pitch medido con densidad `rho_origen`, vistos con `rho_destino`.

    `tipo`: 'Four-Seam', 'Slider', ... (un valor o uno por pitch). `gamma`: dict de cargar_gamma().
    Devuelve (ivb_destino, hb_destino) con la misma forma que las entradas.
    """
    g = cargar_gamma() if gamma is None else gamma
    dr = np.asarray(rho_destino, dtype=float) - np.asarray(rho_origen, dtype=float)
    ivb_d = _mover(ivb, dr, *_coef_de(tipo, "InducedVertBreak", g))
    hb_d = _mover(hb, dr, *_coef_de(tipo, "HorzBreak", g))
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
