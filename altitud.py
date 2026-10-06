"""
altitud.py
Calibra la densidad del aire medida en estadios.py con las altitudes reales
de los 20 estadios de la LMB y estima la altitud de cada estadio.

Idea:
1. La densidad medida (por el frenado de las rectas) tiene una escala
   desconocida: depende del coeficiente de arrastre, la temperatura, etc.
2. Las altitudes reales de los 20 estadios son públicas, pero no sabemos
   qué estadio anónimo es cuál.
3. Dentro de cada categoría de altitud, emparejamos por orden: el estadio con
   el aire menos denso con la altitud más alta. El Harp Helú ya lo conocemos.
4. Ajustamos una recta: densidad medida = a * densidad esperada + b.
5. Con esa recta convertimos la densidad medida de cada estadio en altitud.

Fuentes (altitud en m):
- Mediotiempo / Milenio, 4 de junio de 2025: 13 estadios de altura.
- Open-Meteo (elevación del terreno): Estadio Finsus en Huimilpan, Querétaro
  (20.4244, -100.3013) y los estadios a nivel del mar.
Requiere haber corrido estadios.py antes.
"""
import numpy as np
import pandas as pd

from limpieza import RUTA

HARP_HELU_M = 2232

# Altitudes reales por categoría (orden sin importancia)
ALTITUDES = {
    "Extreme Altitude": [
        2232,  # Harp Helú, Ciudad de México
        2192,  # Hermanos Serdán, Puebla
        2126,  # Finsus, Huimilpan (Querétaro)
        1921,  # Alberto Romo Chávez, Aguascalientes
        1906,  # Francisco Villa, Durango
        1820,  # Domingo Santana, León
    ],
    "Medium Altitude": [
        1566,  # Eduardo Vasconcelos, Oaxaca
        1565,  # Francisco I. Madero, Saltillo
        1555,  # Panamericano, Zapopan (Jalisco)
        1415,  # Monumental, Chihuahua
        1135,  # Revolución, Torreón (Unión Laguna)
    ],
    "No Altitude": [
        607,   # Monclova
        532,   # Monterrey
        127,   # Nuevo Laredo / Laredo
        60,    # Tijuana
        22,    # Villahermosa (Tabasco)
        13,    # Mérida (Yucatán)
        11,    # Cancún (Quintana Roo)
        6,     # Boca del Río (Veracruz)
        3,     # Campeche
    ],
}


def densidad_esperada(h):
    #Densidad del aire a altitud h, relativa al nivel del mar (atmósfera estándar).
    #El exponente 4.25588 es el de densidad; 5.25588 sería el de presión.
    return (1 - 2.25577e-5 * h) ** 4.25588


def altitud_desde_densidad(d):
    #Al revés: de densidad relativa a altitud en metros
    return (1 - d ** (1 / 4.25588)) / 2.25577e-5


#Densidad media por estadio
juegos = pd.read_parquet(RUTA.parent / "estadios.parquet").dropna(subset=["estadio"])
estadios = juegos.groupby("estadio").agg(
    juegos=("year", "size"),
    medida=("densidad_juego", "median"),
    altitud=("altitud", lambda s: s.value_counts().index[0]),
    harp_helu=("harp_helu", "any"),
)

#Matching de ranking a los estadios reales
filas = []
for categoria, alturas in ALTITUDES.items():
    grupo = estadios[estadios["altitud"] == categoria].sort_values("medida")
    alturas = sorted(alturas, reverse=True)
    if categoria == "Extreme Altitude":
        # El Harp Helú ya está identificado: va primero con su altitud real
        grupo = pd.concat([grupo[grupo["harp_helu"]], grupo[~grupo["harp_helu"]]])
        alturas = [HARP_HELU_M] + [h for h in alturas if h != HARP_HELU_M]
    if len(grupo) != len(alturas):
        raise ValueError(f"{categoria}: {len(grupo)} estadios medidos vs {len(alturas)} altitudes")
    for (estadio, fila), h in zip(grupo.iterrows(), alturas):
        filas.append({"estadio": estadio, "categoria": categoria, "juegos": fila["juegos"],
                      "medida": fila["medida"], "altitud_real": h})
tabla = pd.DataFrame(filas).set_index("estadio")
tabla["esperada"] = densidad_esperada(tabla["altitud_real"])

#Ajuste de recta
a, b = np.polyfit(tabla["esperada"], tabla["medida"], 1)
prediccion = a * tabla["esperada"] + b
r2 = 1 - ((tabla["medida"] - prediccion) ** 2).sum() / ((tabla["medida"] - tabla["medida"].mean()) ** 2).sum()

#Estimar cada altura por estadio
tabla["altitud_estimada"] = altitud_desde_densidad((tabla["medida"] - b) / a).round(-1)
tabla["error_m"] = tabla["altitud_estimada"] - tabla["altitud_real"]

print(tabla.round(3).to_string())
print(f"\nRecta: medida = {a:.3f} * esperada + {b:.3f}   R² = {r2:.3f}")
print(f"Error típico: {np.sqrt((tabla['error_m'] ** 2).mean()):.0f} m")

tabla.to_parquet(RUTA.parent / "altitudes.parquet")
print("Guardado en", RUTA.parent / "altitudes.parquet")
