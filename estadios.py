"""
estadios.py
Reconstruye en qué estadio se jugó cada juego, mide la densidad del aire
y marca los juegos jugados en el Harp Helú.

Idea:
1. Pitchers que lanzan en la misma mitad de entrada del mismo juego son del
   mismo equipo. Agrupándolos se reconstruyen los 20 equipos de cada temporada.
2. En la parte alta (Top) lanza el equipo local -> equipo local de cada juego.
3. Cada equipo local juega en su estadio -> estadio de cada juego.
4. El aire frena la bola: k = ay0 / v^2 es proporcional a la densidad del aire.
5. Harp Helú = entre los estadios de altitud extrema, el del equipo local
   con mejor récord (los Diablos).

Resultado: ../estadios.parquet (fuera del repo, porque son datos del ISAC).
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering

from limpieza import RUTA

N_EQUIPOS = 20            # equipos en la LMB
MIN_MITADES_PITCHER = 3   # apariciones mínimas de un pitcher para agruparlo
MIN_RECTAS = 20           # rectas mínimas en un juego para medir su densidad
MIN_JUEGOS_ESTADIO = 60   # juegos mínimos para que un estadio sea candidato
TEMPORADA_BASE = 2025     # temporada que da nombre a los estadios

columnas = ["game_anon_id", "year", "Top/Bottom", "pitcher_anon_id",
            "altitude_category", "AutoPitchType", "vx0", "vy0", "vz0", "ay0",
            "RunsScored"]

df = pd.read_parquet(RUTA, columns=columnas)
df = df.dropna(subset=["pitcher_anon_id"])
df["mitad"] = df["game_anon_id"].astype(str) + "_" + df["Top/Bottom"].astype(str)


def mas_comun(s):
    """Valor más frecuente de una columna (None si está vacía)."""
    return s.value_counts().index[0] if s.notna().any() else None


#Reconstruir equipos de cada temporada 
def equipos_de_temporada(d, year):
    # Tabla pitcher x mitad de juego: 1 si el pitcher lanzó en esa mitad
    tabla = pd.crosstab(d["pitcher_anon_id"], d["mitad"]).clip(upper=1)
    # Ignorar pitchers con muy pocas apariciones: confunden a los equipos
    tabla = tabla[tabla.sum(axis=1) >= MIN_MITADES_PITCHER]

    # Cuántas mitades de juego comparte cada par de pitchers
    comparten = tabla.values @ tabla.values.T
    np.fill_diagonal(comparten, 0)

    # Dividir a los pitchers en 20 equipos
    etiquetas = SpectralClustering(n_clusters=N_EQUIPOS, affinity="precomputed",assign_labels="cluster_qr",random_state=0).fit_predict(comparten)
    return pd.Series([f"{year}_equipo{e:02d}" for e in etiquetas], index=tabla.index)


equipo_de = {}
for year, d in df.groupby("year"):
    for pitcher, equipo in equipos_de_temporada(d, year).items():
        equipo_de[(year, pitcher)] = equipo
df["equipo"] = [equipo_de.get((y, p)) for y, p in zip(df["year"], df["pitcher_anon_id"])]


#Determinar equipo local de cada juego
juegos = df.groupby("game_anon_id").agg(year=("year", "first"),altitud=("altitude_category", mas_comun))
arriba = df[df["Top/Bottom"] == "Top"]   # en la parte alta lanza el local
juegos["local"] = arriba.groupby("game_anon_id")["equipo"].agg(mas_comun)


#Densidad del aire en cada juego
rectas = df[df["AutoPitchType"] == "Four-Seam"].dropna(subset=["vx0", "vy0", "vz0", "ay0"])
k = rectas["ay0"] / (rectas["vx0"]**2 + rectas["vy0"]**2 + rectas["vz0"]**2)
k_juego = k.groupby(rectas["game_anon_id"]).agg(["median", "size"])
juegos["k"] = k_juego["median"].where(k_juego["size"] >= MIN_RECTAS)

# Densidad relativa: 1.00 = mediana de los juegos sin altitud
referencia = juegos.loc[juegos["altitud"] == "No Altitude", "k"].median()
juegos["densidad_juego"] = juegos["k"] / referencia


#Matching de temporadas a estadios
# Cada equipo se empareja con el equipo de 2025 con el que comparte más pitchers, ya que 2025 era el set mas completo
plantel = df.dropna(subset=["equipo"]).groupby("equipo")["pitcher_anon_id"].apply(set)
base = plantel[plantel.index.str.startswith(f"{TEMPORADA_BASE}_")]

estadio_de = {}
for year in sorted(df["year"].unique()):
    otros = plantel[plantel.index.str.startswith(f"{year}_")]
    comun = np.array([[len(a & b) for b in base] for a in otros])
    filas, cols = linear_sum_assignment(-comun)   # emparejamiento 1 a 1
    for f, c in zip(filas, cols):
        estadio_de[otros.index[f]] = "estadio_" + base.index[c][-2:]
juegos["estadio"] = juegos["local"].map(estadio_de)


#Generar tabla de estadios 
carreras = df.groupby(["game_anon_id", "Top/Bottom"])["RunsScored"].sum().unstack()
juegos["gana_local"] = carreras["Bottom"] > carreras["Top"]

tabla = juegos.dropna(subset=["estadio"]).groupby("estadio").agg(
    juegos=("year", "size"),
    densidad=("densidad_juego", "median"),
    altitud=("altitud", mas_comun),
)
pct = juegos.pivot_table(index="estadio", columns="year", values="gana_local", aggfunc="mean")
tabla = tabla.join(pct.add_prefix("pct_local_")).sort_values("densidad")


#Identificar estadio Harp Helu, usando las estadisticas de temporadas pasadas, encontrando el equipo mas dominante de la liga y matching la altura con Diablos Rojos
candidatos = tabla[(tabla["altitud"] == "Extreme Altitude") & (tabla["juegos"] >= MIN_JUEGOS_ESTADIO)]
HARP_HELU = candidatos.filter(like="pct_local").mean(axis=1).idxmax()
juegos["harp_helu"] = (juegos["estadio"] == HARP_HELU) & (juegos["altitud"] == "Extreme Altitude")

print(tabla.round(3).to_string())
print(f"\nHarp Helú = {HARP_HELU} ({juegos['harp_helu'].sum()} juegos)")


#Salida
salida = juegos[["year", "estadio", "altitud", "densidad_juego", "harp_helu"]].reset_index()
salida.to_parquet(RUTA.parent / "estadios.parquet")
print("Guardado en", RUTA.parent / "estadios.parquet")
