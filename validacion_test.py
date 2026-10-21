"""
Paso 9: validación final en el test sellado (2026), con los modelos entrenados SOLO con 2024 y 2025.

Corre ANTES de reentrenar con todo (BEARS_FINAL=1): después, 2026 deja de ser una prueba externa.

Qué mide, todo fuera de muestra:
  1. Stuff+ v1, Stuff+ v3 (oficial), Location+ y Pitching+: mejora de log-loss sobre el baseline por conteo (con las
     frecuencias de 2024-25), en todo 2026 y por separado en pitchers vistos y nuevos; calibración por clase y en Harp Helú.
  2. Deriva de la densidad del aire medida (2026 vs 2024-25) por categoría de altitud.
  3. El ranking de contratación (hecho solo con 2024-25) contra lo que de verdad pasó en 2026 (K−BB% y carreras por turno).

Salida (fuera del repo): ../salidas/validacion_test_2026.json

Uso:
    python validacion_test.py       # necesita todos los modelos de 2024-25 y ranking_contratacion.csv (modo no final)
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from arsenal import agregar_arsenal
from contratacion import MIN_TURNOS, turnos_por_pitcher
from limpieza import cargar_datos
from modelo_location import FEATURES_LOCATION
from modelo_multiclase import CARPETA_MODELOS, CARPETA_SALIDAS, FEATURES, GRUPO, baseline_por_conteo
from modelo_stuff_altitud import preparar
from modelo_stuff_arsenal import FEATURES_V3
from objetivo import CLASES, agregar_resultado
from particion import MODO_FINAL, separar_test
from pitching_plus import FEATURES_PITCHING
from zona import preparar_ubicacion

N = len(CLASES)


def evaluar(nombre, entrenamiento, test, features, modelo_txt):
    """Log-loss del modelo y del baseline por conteo en `test`, y calibración por clase."""
    modelo = lgb.Booster(model_file=str(CARPETA_MODELOS / modelo_txt))
    p = np.asarray(modelo.predict(test[features]), dtype=float)
    base = baseline_por_conteo(entrenamiento["outcome_id"], entrenamiento[["Balls", "Strikes"]], test[["Balls", "Strikes"]])
    y = test["outcome_id"].to_numpy()
    vistos = test[GRUPO].isin(set(entrenamiento[GRUPO])).to_numpy()

    def mejora(m):
        return 100 * (1 - log_loss(y[m], p[m], labels=range(N)) / log_loss(y[m], base[m], labels=range(N)))

    todos = np.ones(len(test), dtype=bool)
    por_clase = {}
    for k, c in enumerate(CLASES):
        yk = (y == k).astype(int)
        por_clase[c] = float(100 * (1 - log_loss(yk, p[:, k], labels=[0, 1]) / log_loss(yk, base[:, k], labels=[0, 1])))
    res = {"modelo": nombre, "n": int(len(test)), "pitchers": int(test[GRUPO].nunique()),
           "mejora_%": float(mejora(todos)), "mejora_vistos_%": float(mejora(vistos)), "mejora_nuevos_%": float(mejora(~vistos)),
           "n_vistos": int(vistos.sum()), "n_nuevos": int((~vistos).sum()), "mejora_por_clase_%": por_clase,
           "calibracion_pp": {c: float(100 * (p[:, k].mean() - (y == k).mean())) for k, c in enumerate(CLASES)}}
    if "harp_helu" in test.columns and test["harp_helu"].any():
        h = test["harp_helu"].to_numpy()
        res["harp_helu"] = {"n": int(h.sum()), "real_%": {c: float(100 * (y[h] == k).mean()) for k, c in enumerate(CLASES)},
                            "predicho_%": {c: float(100 * p[h, k].mean()) for k, c in enumerate(CLASES)}}
    print(f"{nombre:34s} n={res['n']:>7,} | mejora {res['mejora_%']:.2f} % (vistos {res['mejora_vistos_%']:.2f} %, nuevos {res['mejora_nuevos_%']:.2f} %)")
    return res


def deriva_densidad():
    est = pd.read_parquet(CARPETA_SALIDAS.parent / "estadios.parquet")
    t = est.groupby(["altitud", "year"])["densidad_juego"].agg(["mean", "size"]).round(3).unstack("year")
    harp = est[est["harp_helu"]].groupby("year")["densidad_juego"].mean().round(4)
    print("\nDensidad media del aire por categoría de altitud y temporada:")
    print(t.to_string())
    print("Harp Helú:", harp.to_dict())
    return {"por_altitud": {f"{a}_{y}": float(t.loc[a, ("mean", y)]) for a in t.index for y in t["mean"].columns},
            "harp": {int(y): float(v) for y, v in harp.items()}}


def validar_ranking():
    """Ranking hecho con 2024+2025 (ranking_contratacion.csv) contra los resultados reales de 2026."""
    r = pd.read_csv(CARPETA_SALIDAS / "ranking_contratacion.csv").set_index("pitcher_anon_id")
    res26 = turnos_por_pitcher(incluir_test=True)
    ultimo = res26["year"].max()
    r26 = res26[res26["year"] == ultimo].set_index("pitcher_anon_id")
    w = r[["rating_100", "ejecucion_100", "kbb", "runs_pa", "n_pitches"]].join(r26[["turnos", "kbb", "runs_pa"]].add_suffix("_26"), how="inner")
    w = w[w["turnos_26"] >= MIN_TURNOS].dropna()
    n = len(w)
    out = {"n_pitchers": int(n), "temporada": int(ultimo)}
    for meta, nombre in [("kbb", "K−BB%"), ("runs_pa", "carreras por turno")]:
        y = w[f"{meta}_26"]
        r_rank, r_ej, r_pasado = (float(w[c].corr(y)) for c in ["rating_100", "ejecucion_100", meta])
        top = w["rating_100"].nlargest(20).index
        z = np.arctanh(r_rank)
        out[meta] = {"nombre": nombre, "r_rating": r_rank, "r_ejecucion": r_ej, "r_resultado_pasado": r_pasado,
                     "r_rating_ic95": [float(np.tanh(z - 1.96 / np.sqrt(n - 3))), float(np.tanh(z + 1.96 / np.sqrt(n - 3)))],
                     "top20": float(y.loc[top].mean()), "todos": float(y.mean()),
                     "peor20": float(y.loc[w["rating_100"].nsmallest(20).index].mean())}
        print(f"  {nombre:20s} r rating→{ultimo} = {r_rank:.2f} (IC95 {out[meta]['r_rating_ic95'][0]:.2f} a {out[meta]['r_rating_ic95'][1]:.2f}) | "
              f"resultado pasado r = {r_pasado:.2f} | top 20 {100 * out[meta]['top20']:+.2f} vs todos {100 * out[meta]['todos']:+.2f} vs peor 20 {100 * out[meta]['peor20']:+.2f}")
    return out


def main():
    if MODO_FINAL:
        raise SystemExit("validacion_test.py debe correr SIN BEARS_FINAL=1 (con 2026 sellado) y con los modelos de 2024-25.")
    datos = agregar_resultado(cargar_datos())
    ent_v1, test_v1 = separar_test(datos)
    print(f"Test sellado: temporada {test_v1['year'].max()} | entrenamiento {len(ent_v1):,} | test {len(test_v1):,} lanzamientos\n")
    resultados = {"modelos": []}
    resultados["modelos"].append(evaluar("Stuff+ v1 (sin altitud)", ent_v1, test_v1, FEATURES, "modelo_resultado.txt"))

    ent_v3, test_v3 = separar_test(agregar_arsenal(preparar(datos)))
    resultados["modelos"].append(evaluar("Stuff+ v3 (oficial)", ent_v3, test_v3, FEATURES_V3, "modelo_stuff_arsenal.txt"))

    ent_u, test_u = separar_test(preparar_ubicacion(datos))
    resultados["modelos"].append(evaluar("Location+", ent_u, test_u, FEATURES_LOCATION, "modelo_location.txt"))
    resultados["modelos"].append(evaluar("Pitching+", ent_u, test_u, FEATURES_PITCHING, "modelo_pitching.txt"))

    resultados["densidad"] = deriva_densidad()
    print("\nRanking de contratación (2024+2025) contra los resultados reales de 2026:")
    resultados["ranking"] = validar_ranking()

    ruta = CARPETA_SALIDAS / "validacion_test_2026.json"
    ruta.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado: {ruta}")


if __name__ == "__main__":
    main()
