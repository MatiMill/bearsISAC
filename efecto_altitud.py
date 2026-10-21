"""
Altitud, Idea 1: ¿cuánto cambian el movimiento y la velocidad con la densidad del aire en la LMB?

Para cada tipo de pitch se mide el efecto DENTRO del mismo pitcher (efecto fijo pitcher × tipo),
así el talento del pitcher y el equipo no se confunden con el estadio:

    x_{i,t} = alpha_i + gamma * rho_t + error      x = IVB, HB, SpinRate o RelSpeed del pitch t
                                                   rho = densidad_juego del juego (1.00 = nivel del mar)

Se estima con la transformación "within" (restar la media de cada pitcher × tipo). Como la física
dice que el movimiento por spin es proporcional a rho (Magnus), se reporta la ELASTICIDAD

    beta = gamma * mean(rho) / mean(x)     (beta = 1: el movimiento es proporcional a rho;
                                            beta = 0: la altitud no lo cambia)

y el cambio en las unidades de x al pasar de nivel del mar (1.00) a Harp Helú (HARP_RHO).

Además de ese gamma único por tipo (aditivo), se ajusta un segundo modelo donde el cambio crece con cuánto
se mueve el pitcher (auditoría 2026-10-21: un solo gamma mueve en sentido contrario a los pitches atípicos):

    x_{i,t} = alpha_i + (g0 + kappa * xm_i) * rho_t + error      xm_i = movimiento medio del pitcher × tipo

`g0` y `kappa` son los que usa altitud_movimiento.py (forma exponencial, exactamente invertible).
Intervalos de confianza al 95 % por bootstrap de PITCHERS (300 remuestreos): los pitches de un
mismo pitcher no son independientes.

SpinRate y RelSpeed son controles: el aire no debería cambiar el spin ni la velocidad de
salida (beta ≈ 0). Si salieran ≠ 0, el efecto de IVB/HB estaría contaminado por otra cosa.

Solo temporadas de entrenamiento (la última temporada es el test sellado).

Salida (fuera del repo): ../salidas/efecto_altitud.csv
    tipo, variable, n_pitches, n_pitchers, gamma, beta, beta_lo, beta_hi, delta_harp, delta_lo, delta_hi,
    g0, kappa, kappa_lo, kappa_hi

Uso:
    python efecto_altitud.py
"""
import numpy as np
import pandas as pd

from limpieza import cargar_datos
from altitud_movimiento import RHO_HARP as HARP_RHO     # densidad media de los juegos en Harp Helú (una sola fuente)
from modelo_multiclase import CARPETA_SALIDAS, GRUPO, SEMILLA
from particion import separar_test

VARIABLES = ["InducedVertBreak", "HorzBreak", "SpinRate", "RelSpeed"]
MIN_PITCHES_CELDA = 20      # pitches mínimos de un pitcher × tipo para entrar
N_BOOT = 300
RUTA_ESTADIOS = CARPETA_SALIDAS.parent / "estadios.parquet"


def agregar_densidad(df):
    """Agrega `densidad_juego` y `harp_helu` (de ../estadios.parquet) a los pitches de df.

    Unos pocos juegos no tienen densidad medida (2 en el entrenamiento, 19 pitches; 5 más en 2026): se les
    pone la densidad media de su estadio y, si tampoco tiene estadio (1 juego de 2026), la media de su categoría de
    altitud. Si algún juego no está en estadios.parquet, falla.
    """
    est = pd.read_parquet(RUTA_ESTADIOS)[["game_anon_id", "estadio", "altitud", "densidad_juego", "harp_helu"]].copy()
    rho = est["densidad_juego"]
    rho = rho.fillna(est.groupby("estadio")["densidad_juego"].transform("mean"))
    est["densidad_juego"] = rho.fillna(est.groupby("altitud")["densidad_juego"].transform("mean"))
    out = df.merge(est[["game_anon_id", "densidad_juego", "harp_helu"]], on="game_anon_id", how="left",
                   validate="many_to_one")
    assert out["densidad_juego"].notna().all(), "hay juegos sin densidad en estadios.parquet"
    out["harp_helu"] = out["harp_helu"].astype(bool)
    return out


def cargar_base():
    """Pitches de entrenamiento con la densidad de su juego."""
    return agregar_densidad(separar_test(cargar_datos())[0])


def sumas_por_pitcher(df, var):
    """Sumas within por pitcher (para el bootstrap), una fila por pitcher:
    xy, rr, n, sum_x, sum_rho, y las que piden g0 y kappa: mxy = Σ xm·rd·xd, mrr = Σ xm·rd², mmrr = Σ xm²·rd²."""
    celdas = df.groupby([GRUPO, "AutoPitchType"], observed=True)
    d = df[[GRUPO, "AutoPitchType", var, "densidad_juego"]].copy()
    d["n_celda"] = celdas[var].transform("size")
    d = d[d["n_celda"] >= MIN_PITCHES_CELDA]
    celdas = d.groupby([GRUPO, "AutoPitchType"], observed=True)
    d["xd"] = d[var] - celdas[var].transform("mean")
    d["rd"] = d["densidad_juego"] - celdas["densidad_juego"].transform("mean")
    d["xm"] = celdas[var].transform("mean")
    d["xy"] = d["xd"] * d["rd"]
    d["rr"] = d["rd"] ** 2
    d["mxy"] = d["xm"] * d["xy"]
    d["mrr"] = d["xm"] * d["rr"]
    d["mmrr"] = d["xm"] ** 2 * d["rr"]
    d["x"] = d[var]
    d["rho"] = d["densidad_juego"]
    d["uno"] = 1.0
    return d.groupby(GRUPO)[["xy", "rr", "uno", "x", "rho", "mxy", "mrr", "mmrr"]].sum()


def estimar(s):
    """gamma, beta, delta, g0 y kappa a partir de las sumas (array 2D: xy, rr, n, x, rho, mxy, mrr, mmrr).

    g0 y kappa: mínimos cuadrados de xd = (g0 + kappa·xm)·rd, con las ecuaciones normales
    [Σrr Σmrr; Σmrr Σmmrr]·[g0, kappa] = [Σxy, Σmxy].
    """
    xy, rr, n, sx, sr, mxy, mrr, mmrr = s.sum(axis=0)
    gamma = xy / rr
    media_x, media_rho = sx / n, sr / n
    g0, kappa = np.linalg.lstsq(np.array([[rr, mrr], [mrr, mmrr]]), np.array([xy, mxy]), rcond=None)[0]
    return gamma, gamma * media_rho / media_x, gamma * (HARP_RHO - 1.0), g0, kappa


def main():
    rng = np.random.default_rng(SEMILLA)
    df = cargar_base()
    print(f"Pitches de entrenamiento: {len(df):,} | juegos: {df['game_anon_id'].nunique():,} | "
          f"densidad: media {df['densidad_juego'].mean():.3f}, rango "
          f"[{df['densidad_juego'].min():.2f}, {df['densidad_juego'].max():.2f}]\n")

    filas = []
    for tipo, sub in [("TODOS", df)] + [(t, g) for t, g in df.groupby("AutoPitchType", observed=True)]:
        for var in VARIABLES:
            s = sumas_por_pitcher(sub, var)
            if len(s) < 30:
                continue
            arr = s.to_numpy()
            gamma, beta, delta, g0, kappa = estimar(arr)
            boot = np.array([estimar(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(N_BOOT)])
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            filas.append({"tipo": tipo, "variable": var, "n_pitches": int(arr[:, 2].sum()), "n_pitchers": len(s),
                          "media_x": arr[:, 3].sum() / arr[:, 2].sum(),
                          "gamma": gamma, "beta": beta, "beta_lo": lo[1], "beta_hi": hi[1],
                          "delta_harp": delta, "delta_lo": lo[2], "delta_hi": hi[2],
                          "g0": g0, "kappa": kappa, "kappa_lo": lo[4], "kappa_hi": hi[4]})
    tabla = pd.DataFrame(filas)

    CARPETA_SALIDAS.mkdir(exist_ok=True)
    tabla.to_csv(CARPETA_SALIDAS / "efecto_altitud.csv", index=False)

    nombres = {"InducedVertBreak": "IVB", "HorzBreak": "HB", "SpinRate": "SpinRate", "RelSpeed": "RelSpeed"}
    for var in VARIABLES:
        t = tabla[tabla["variable"] == var].copy()
        t["beta (IC 95%)"] = t.apply(lambda r: f"{r.beta:5.2f} [{r.beta_lo:5.2f}, {r.beta_hi:5.2f}]", axis=1)
        t["Δ en Harp Helú vs mar (IC)"] = t.apply(
            lambda r: f"{r.delta_harp:6.2f} [{r.delta_lo:6.2f}, {r.delta_hi:6.2f}]", axis=1)
        print(f"--- {nombres[var]} (media del tipo = media_x; Δ en unidades de la variable) ---")
        t["g0, kappa (IC)"] = t.apply(lambda r: f"{r.g0:6.2f}, {r.kappa:5.2f} [{r.kappa_lo:5.2f}, {r.kappa_hi:5.2f}]", axis=1)
        print(t[["tipo", "n_pitches", "n_pitchers", "media_x", "beta (IC 95%)", "Δ en Harp Helú vs mar (IC)", "g0, kappa (IC)"]]
              .round(2).to_string(index=False))
        print()
    print(f"Guardado: {CARPETA_SALIDAS / 'efecto_altitud.csv'}")


if __name__ == "__main__":
    main()
