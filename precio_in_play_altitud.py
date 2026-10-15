"""
Altitud, Idea 3 (versión simple): ¿cuánto más vale un batazo en juego cuando el aire es menos denso?

En el aire de Harp Helú la bola bateada vuela más, así que el mismo batazo en juego cuesta más carreras al
pitcher. Los precios RV_k de run_values.py son un solo promedio de toda la liga; aquí se mide cuánto se
desvía el valor de los turnos que terminan en juego con la densidad del juego:

    valor_turno = alpha_bateador + delta * rho + efectos de conteo + efectos del estado (outs, corredores) + error

- Efecto fijo por BATEADOR: se compara al mismo bateador en aires distintos (no a los bateadores de los
  Diablos contra los de otros equipos).
- Errores estándar agrupados por bateador.
- `valor_turno` es el mismo de run_values.py: RE(estado al terminar) − RE(estado al empezar) + carreras.
- Solo temporadas de entrenamiento (la última temporada es el test sellado).

Uso en el score: el precio de in_play pasa de RV_inplay(b, s) a RV_inplay(b, s) + delta · (rho − rho_medio),
con rho_medio = densidad media de los pitches de entrenamiento. Solo cambia el precio de in_play: los otros
resultados no dependen del aire.

Límite: es la versión simple (un solo delta para todos los batazos). El modelo de contacto (paso 7) lo
afinaría por tipo de contacto (rolling vs. elevado).

Salida (fuera del repo): ../salidas/precio_in_play_altitud.csv  (delta, se, rho_medio, n, y cambio en Harp Helú)

Uso:
    python precio_in_play_altitud.py
"""
import numpy as np
import pandas as pd

from altitud_movimiento import RHO_HARP, RHO_MAR
from efecto_altitud import agregar_densidad
from modelo_multiclase import CARPETA_SALIDAS
from run_values import armar_turnos, carreras_esperadas, cargar_crudos, valor_de_turnos

RUTA_SALIDA = CARPETA_SALIDAS / "precio_in_play_altitud.csv"


def turnos_en_juego():
    """Turnos que terminaron en juego, con valor, conteo, estado, bateador y densidad del juego."""
    df, _ = cargar_crudos()
    turnos = armar_turnos(df)
    turnos = valor_de_turnos(df, turnos, carreras_esperadas(turnos))
    primero = df.groupby("turno").agg(game_anon_id=("game_anon_id", "first"), batter=("batter_anon_id", "first"))
    t = turnos.join(primero)
    t = t[t["terminado"] & (t["pitch_final"] == "InPlay")].copy()
    t = agregar_densidad(t.reset_index())
    columnas = ["valor", "balls_final", "strikes_final", "outs", "corredores", "densidad_juego", "batter"]
    sin_dato = t[columnas].isna().any(axis=1)
    if sin_dato.any():
        print(f"(se descartan {int(sin_dato.sum())} turnos con datos vacíos en {columnas})")
    return t[~sin_dato]


def regresion_con_efecto_fijo(t):
    """delta (valor por 1.00 de densidad) con efecto fijo por bateador y errores agrupados por bateador."""
    X = pd.concat([
        t[["densidad_juego"]].astype(float),
        pd.get_dummies(t["balls_final"].astype(int).astype(str) + "-" + t["strikes_final"].astype(int).astype(str),
                       prefix="c", drop_first=True),
        pd.get_dummies(t["outs"].astype(int).astype(str) + "-" + t["corredores"].astype(int).astype(str),
                       prefix="e", drop_first=True),
    ], axis=1).astype(float)
    y = t["valor"].astype(float)
    g = t["batter"].to_numpy()
    # transformación within: restar la media de cada bateador
    Xd = X - X.groupby(g).transform("mean")
    yd = y - y.groupby(g).transform("mean")
    A = Xd.to_numpy()
    beta, *_ = np.linalg.lstsq(A, yd.to_numpy(), rcond=None)
    e = yd.to_numpy() - A @ beta
    bread = np.linalg.pinv(A.T @ A)
    scores = pd.DataFrame(A * e[:, None]).groupby(g).sum().to_numpy()       # X'e por bateador
    meat = scores.T @ scores
    n_g, n, k = len(scores), len(A), A.shape[1]
    ajuste = n_g / (n_g - 1) * (n - 1) / (n - k - n_g)                       # corrección de muestras finitas
    cov = ajuste * bread @ meat @ bread
    return float(beta[0]), float(np.sqrt(cov[0, 0])), n_g


def main():
    t = turnos_en_juego()
    rho_medio = float(t["densidad_juego"].mean())
    print(f"Turnos que terminaron en juego (entrenamiento): {len(t):,} | bateadores: {t['batter'].nunique():,} | "
          f"densidad media {rho_medio:.3f}")

    print("\nValor promedio de un turno que termina en juego (carreras), sin controles:")
    print(t.groupby("harp_helu")["valor"].agg(["mean", "size"]).rename(index={True: "Harp Helú", False: "resto"})
          .round(4).to_string())

    delta, se, n_bateadores = regresion_con_efecto_fijo(t)
    cambio = delta * (RHO_HARP - RHO_MAR)
    print(f"\nEfecto de la densidad sobre el valor de un batazo en juego (efecto fijo por bateador, {n_bateadores:,} bateadores):")
    print(f"   delta = {delta:+.4f} carreras por 1.00 de densidad (EE {se:.4f}; IC 95 % "
          f"[{delta - 1.96 * se:+.4f}, {delta + 1.96 * se:+.4f}])")
    print(f"   Harp Helú (rho = {RHO_HARP}) vs nivel del mar (1.00): {cambio:+.4f} carreras por batazo en juego "
          f"[{(delta + 1.96 * se) * (RHO_HARP - RHO_MAR):+.4f}, {(delta - 1.96 * se) * (RHO_HARP - RHO_MAR):+.4f}]")
    print(f"   (en perspectiva: el precio de un batazo en juego sin ajustar es ≈ {t['valor'].mean():+.3f} carreras en promedio)")

    CARPETA_SALIDAS.mkdir(exist_ok=True)
    pd.DataFrame([{"delta": delta, "se": se, "rho_medio": rho_medio, "n_turnos": len(t), "n_bateadores": n_bateadores,
                   "cambio_harp_vs_mar": cambio}]).to_csv(RUTA_SALIDA, index=False)
    print(f"\nGuardado: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
