"""
Altitud, Idea 2 (contrafactual): el MISMO pitch en el aire de nivel del mar y en el de Harp Helú.

Stuff+ v3 (altitud + arsenal) califica cada pitch, con su movimiento llevado a nivel del mar, en dos escenarios:

    mar    movimiento normalizado, densidad 1.00
    harp   movimiento ajustado a Harp Helú (altitud_movimiento.ajustar_movimiento), densidad 0.828; en v3 también
           cambian ivb_diff_vs_fb y hb_diff_vs_fb, porque la recta de referencia pierde movimiento en el mismo aire

y la diferencia se separa en tres canales:

    movimiento   la bola se mueve menos         (mismo modelo, densidad 1.00, solo cambia IVB/HB)
    resultado    lo que el modelo aprendió de la densidad (whiff, foul, bola en juego), con el movimiento ya ajustado
    precio       un batazo en juego cuesta más carreras (precio_in_play_altitud.py)

xRV de cada escenario = promedio de los 12 conteos con los pesos de la liga (igual que Stuff+ v1, scoring.py),
con el precio de in_play desplazado: RV_inplay + delta * (rho - rho_medio) * P(in_play).
Escala: Stuff+ = 100 + 10 z, con media y sd del escenario `mar` POR TEMPORADA; `harp` usa los mismos mu y sigma,
así la diferencia está en puntos de Stuff+ comparables.

Honestidad (out-of-fold): cada pitch se predice con el modelo de fold que no vio a su pitcher.

Salidas (fuera del repo):
    ../salidas/scores_stuff_altitud_pitch.parquet
    ../salidas/delta_altitud_tipo.csv, delta_altitud_pitcher_tipo.csv, delta_altitud_pitcher.csv,
    ../salidas/calibracion_stuff_altitud.json (media, sd y precio de la bola en juego)

Uso:
    python contrafactual_altitud.py
    # antes: efecto_altitud.py, precio_in_play_altitud.py, modelo_stuff_altitud.py y modelo_stuff_arsenal.py
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from altitud_movimiento import RHO_HARP, RHO_MAR, ajustar_movimiento
from arsenal import diferencias_en_aire
from modelo_multiclase import CARPETA_MODELOS, CARPETA_SALIDAS, GRUPO, N_PARTES
from modelo_stuff_altitud import leer_folds
from modelo_stuff_arsenal import FEATURES_V3, cargar_entrenamiento_arsenal
from objetivo import CLASES
from scoring import COLS_PROB, COLS_RV, cargar_rv

CONTEOS = [(b, s) for b in (0, 1, 2, 3) for s in (0, 1, 2)]
I_IN_PLAY = CLASES.index("in_play")
RUTA_CALIBRACION = CARPETA_SALIDAS / "calibracion_stuff_altitud.json"
FEATURES = FEATURES_V3
PREFIJO_MODELO = "modelo_stuff_arsenal"
VERSION = "v3"


def xrv_neutral(predecir, base, features, rv_wide, pesos, rho=None):
    """xRV neutral al conteo (12 conteos, pesos de la liga) y exposición a bola en juego, por pitch.

    `predecir`: función (DataFrame de features) -> probabilidades (n, 5). `base`: DataFrame con todas las
    features menos Balls y Strikes (y menos densidad_juego si se pasa `rho`). `rho`: densidad escalar a fijar.
    Devuelve (xrv_base, exposicion_in_play): xRV con los precios de run_values.py y sum_c w_c P(in_play | c).
    """
    rv = rv_wide.set_index(["Balls", "Strikes"])[COLS_RV]
    w = pesos.set_index(["Balls", "Strikes"])["peso"]
    X = base.copy()
    if rho is not None:
        X["densidad_juego"] = float(rho)
    xrv = np.zeros(len(base))
    exposicion = np.zeros(len(base))
    for b, s in CONTEOS:
        X["Balls"], X["Strikes"] = float(b), float(s)
        p = np.asarray(predecir(X[features]), dtype=float)
        xrv += w.loc[(b, s)] * (p * rv.loc[(b, s)].to_numpy()).sum(axis=1)
        exposicion += w.loc[(b, s)] * p[:, I_IN_PLAY]
    return xrv, exposicion


def desplazamiento_precio(exposicion, rho, delta, rho_medio):
    """Carreras que se suman al xRV por el precio de in_play en un aire de densidad `rho`."""
    return delta * (rho - rho_medio) * exposicion


def base_escenario(df, rho_destino):
    """Features físicas del escenario: movimiento llevado de nivel del mar (IVB_n, HB_n) a `rho_destino`."""
    base = df[["RelSpeed", "SpinRate", "Extension", "RelHeight", "RelSide", "SpinAxis"]].copy()
    if rho_destino == RHO_MAR:
        base["IVB_n"], base["HB_n"] = df["IVB_n"].to_numpy(), df["HB_n"].to_numpy()
    else:
        base["IVB_n"], base["HB_n"] = ajustar_movimiento(df["IVB_n"], df["HB_n"], RHO_MAR, rho_destino,
                                                         tipo=df["AutoPitchType"].astype(str))
    base["velo_diff_vs_fb"] = df["velo_diff_vs_fb"].to_numpy()
    base["release_diff_vs_fb"] = df["release_diff_vs_fb"].to_numpy()
    base["pitch_usage_pct"] = df["pitch_usage_pct"].to_numpy()
    if rho_destino == RHO_MAR:
        base["ivb_diff_vs_fb"], base["hb_diff_vs_fb"] = df["ivb_diff_vs_fb"].to_numpy(), df["hb_diff_vs_fb"].to_numpy()
    else:
        base["ivb_diff_vs_fb"], base["hb_diff_vs_fb"] = diferencias_en_aire(df, rho_destino)
    return base


def main():
    df = cargar_entrenamiento_arsenal()
    fold = df[GRUPO].map(leer_folds()).to_numpy()
    rv_wide, pesos = cargar_rv()
    ip = pd.read_csv(CARPETA_SALIDAS / "precio_in_play_altitud.csv").iloc[0]
    delta, rho_medio = float(ip["delta"]), float(ip["rho_medio"])
    print(f"Pitches: {len(df):,} | precio in_play: delta = {delta:+.4f} carreras por 1.00 de densidad, rho medio {rho_medio:.3f}\n")

    cols = ["xrv_mar", "ip_mar", "xrv_harp_mov", "xrv_harp", "ip_harp"]
    res = np.zeros((len(df), len(cols)))
    for k in range(1, N_PARTES + 1):
        idx = np.flatnonzero(fold == k)
        sub = df.iloc[idx]
        modelo = lgb.Booster(model_file=str(CARPETA_MODELOS / f"{PREFIJO_MODELO}_fold{k}.txt"))
        b_mar, b_harp = base_escenario(sub, RHO_MAR), base_escenario(sub, RHO_HARP)
        res[idx, 0], res[idx, 1] = xrv_neutral(modelo.predict, b_mar, FEATURES, rv_wide, pesos, rho=RHO_MAR)
        res[idx, 2], _ = xrv_neutral(modelo.predict, b_harp, FEATURES, rv_wide, pesos, rho=RHO_MAR)
        res[idx, 3], res[idx, 4] = xrv_neutral(modelo.predict, b_harp, FEATURES, rv_wide, pesos, rho=RHO_HARP)
        print(f"  fold {k}: {len(idx):,} pitches × 12 conteos × 3 escenarios")
    r = pd.DataFrame(res, columns=cols, index=df.index)

    # xRV total de cada escenario = xRV de run_values + desplazamiento del precio de in_play
    xrv_mar = r["xrv_mar"] + desplazamiento_precio(r["ip_mar"], RHO_MAR, delta, rho_medio)
    xrv_harp = r["xrv_harp"] + desplazamiento_precio(r["ip_harp"], RHO_HARP, delta, rho_medio)
    s_mar = -xrv_mar
    mu = s_mar.groupby(df["year"]).transform("mean")
    sd = s_mar.groupby(df["year"]).transform("std")

    def a_puntos(dif_xrv):      # diferencia de xRV (carreras) -> puntos de Stuff+ (+ = mejor para el pitcher)
        return -10 * dif_xrv / sd

    out = df[["PitchUID", GRUPO, "AutoPitchType", "year", "harp_helu", "densidad_juego"]].copy()
    out["stuff_plus_mar"] = 100 + 10 * (s_mar - mu) / sd
    out["stuff_plus_harp"] = 100 + 10 * (-xrv_harp - mu) / sd
    out["delta_total"] = out["stuff_plus_harp"] - out["stuff_plus_mar"]
    out["delta_movimiento"] = a_puntos(r["xrv_harp_mov"] - r["xrv_mar"])
    out["delta_resultado"] = a_puntos(r["xrv_harp"] - r["xrv_harp_mov"])
    out["delta_precio"] = a_puntos(desplazamiento_precio(r["ip_harp"], RHO_HARP, delta, rho_medio)
                                   - desplazamiento_precio(r["ip_mar"], RHO_MAR, delta, rho_medio))
    # El precio de la bola en juego mueve a TODOS los pitches casi igual (efecto de parque). Lo que distingue a un
    # pitch de otro es cuánto más o menos pierde que el promedio: delta_relativo = delta_total - media de la liga.
    delta_medio = float(out["delta_total"].mean())
    out["delta_relativo"] = out["delta_total"] - delta_medio
    suma = out[["delta_movimiento", "delta_resultado", "delta_precio"]].sum(axis=1)
    assert np.allclose(suma, out["delta_total"], atol=1e-6), "los tres canales no suman el delta total"
    out["stuff_plus_v1"] = out["PitchUID"].map(
        pd.read_parquet(CARPETA_SALIDAS / "scores_pitch.parquet").set_index("PitchUID")["stuff_plus"])

    # ---- guardar
    CARPETA_SALIDAS.mkdir(exist_ok=True)
    out.to_parquet(CARPETA_SALIDAS / "scores_stuff_altitud_pitch.parquet", index=False)
    canales = ["delta_movimiento", "delta_resultado", "delta_precio", "delta_total", "delta_relativo"]
    por_tipo = (out.groupby("AutoPitchType", observed=True)
                .agg(n=("delta_total", "size"), stuff_plus_mar=("stuff_plus_mar", "mean"),
                     stuff_plus_harp=("stuff_plus_harp", "mean"), **{c: (c, "mean") for c in canales}))
    por_tipo.to_csv(CARPETA_SALIDAS / "delta_altitud_tipo.csv")
    por_pitcher = (out.groupby(GRUPO).agg(n=("delta_total", "size"), stuff_plus_mar=("stuff_plus_mar", "mean"),
                                           stuff_plus_harp=("stuff_plus_harp", "mean"), delta_total=("delta_total", "mean"),
                                           delta_relativo=("delta_relativo", "mean")))
    por_pitcher.to_csv(CARPETA_SALIDAS / "delta_altitud_pitcher.csv")
    (out.groupby([GRUPO, "AutoPitchType"], observed=True)
     .agg(n=("delta_total", "size"), stuff_plus_mar=("stuff_plus_mar", "mean"), stuff_plus_harp=("stuff_plus_harp", "mean"),
          delta_total=("delta_total", "mean"), delta_relativo=("delta_relativo", "mean"))
     .to_csv(CARPETA_SALIDAS / "delta_altitud_pitcher_tipo.csv"))
    anio = out["year"].astype(int)
    calibracion = {"mu_por_anio": {int(y): float(v) for y, v in s_mar.groupby(anio).mean().items()},
                   "sd_por_anio": {int(y): float(v) for y, v in s_mar.groupby(anio).std().items()},
                   "mu": float(s_mar.mean()), "sd": float(s_mar.std()),
                   "delta_total_medio": delta_medio, "delta_in_play": delta, "rho_medio": rho_medio, "rho_harp": RHO_HARP, "rho_mar": RHO_MAR}
    RUTA_CALIBRACION.write_text(json.dumps(calibracion, indent=2), encoding="utf-8")

    # ---- Sanity 1: media 100 / sd 10 del escenario mar por temporada
    print(f"\nSanity 1 - Stuff+ {VERSION} a nivel del mar por temporada (media 100, sd 10 por construcción):")
    print(out.groupby("year")["stuff_plus_mar"].agg(["mean", "std", "count"]).round(2).to_string())

    # ---- Sanity 2: el sesgo de v1 (Harp Helú ≈ 97 vs ≈ 100) desaparece con la normalización
    print("\nSanity 2 - Stuff+ medio de los pitches lanzados en Harp Helú vs el resto:")
    t = out.groupby("harp_helu")[["stuff_plus_v1", "stuff_plus_mar"]].mean().rename(index={True: "Harp Helú", False: "resto"})
    t[f"diferencia v1 vs {VERSION}"] = t["stuff_plus_mar"] - t["stuff_plus_v1"]
    print(t.round(2).to_string())
    print(f"   correlación Stuff+ v1 vs {VERSION} (pitch): r = {out['stuff_plus_v1'].corr(out['stuff_plus_mar']):.3f}")
    pv = out.groupby(GRUPO).agg(v1=("stuff_plus_v1", "mean"), v2=("stuff_plus_mar", "mean"), n=("stuff_plus_mar", "size"))
    pv = pv[pv["n"] >= 200]
    print(f"   correlación a nivel pitcher (n ≥ 200, {len(pv)} pitchers): r = {pv['v1'].corr(pv['v2']):.3f}")

    # ---- Resultado: delta de Stuff+ en Harp Helú vs nivel del mar, por tipo de pitch y por canal
    print("\nDelta de Stuff+ en Harp Helú vs nivel del mar, por tipo de pitch (puntos; + = mejor para el pitcher):")
    print(por_tipo.round(2).to_string())
    print("\n(delta_relativo = delta_total menos el delta promedio de la liga: cuánto MÁS o MENOS pierde ese tipo que un pitch promedio)")
    print(f"\nTodos los pitches: delta total {out['delta_total'].mean():+.2f} "
          f"(movimiento {out['delta_movimiento'].mean():+.2f}, resultado {out['delta_resultado'].mean():+.2f}, "
          f"precio de la bola en juego {out['delta_precio'].mean():+.2f})")
    print(f"Dispersión entre pitchers (n ≥ 200): delta total sd = {por_pitcher.loc[por_pitcher['n'] >= 200, 'delta_total'].std():.2f}, "
          f"rango [{por_pitcher.loc[por_pitcher['n'] >= 200, 'delta_total'].min():+.1f}, "
          f"{por_pitcher.loc[por_pitcher['n'] >= 200, 'delta_total'].max():+.1f}]")
    print(f"\nGuardado en {CARPETA_SALIDAS}: scores_stuff_altitud_pitch.parquet, delta_altitud_tipo.csv, "
          f"delta_altitud_pitcher_tipo.csv, delta_altitud_pitcher.csv, calibracion_stuff_altitud.json")


if __name__ == "__main__":
    main()
