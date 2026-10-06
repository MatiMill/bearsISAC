"""
zona.py
Ubicación del pitch respecto al bateador y la zona de strike en 9 cuadrantes.

- lado_relativo: PlateLocSide visto desde el bateador. Positivo = adentro (hacia el
  bateador), negativo = afuera, sin importar si batea a la derecha o a la izquierda.
  Se comprueba con los pelotazos (HitByPitch): sin voltear, están en +2.0 ft para
  derechos y -2.0 ft para zurdos; volteando a los zurdos, ambos quedan en +2.0.

- zona: la zona de strike del dataset es el rectángulo |PlateLocSide| <= 0.7083 ft
  (17" de home / 2) y 1.62 <= PlateLocHeight <= 3.21 ft (coincide con
  in_strike_zone en el 99.999 % de los pitches). Se divide en 3 x 3:

        alto      1 adentro   2 centro   3 afuera
        medio     4 adentro   5 centro   6 afuera
        bajo      7 adentro   8 centro   9 afuera
        fuera de la zona: 10 (una sola región de "bola", como pide el brief)

  El modelo usa las coordenadas continuas; la zona sirve para reportes y el dashboard.

Uso:
    python zona.py      # revisiones sobre los datos limpios
"""
import numpy as np
import pandas as pd

from limpieza import TIPOS_VALIDOS

ZONA_LADO = 0.7083     # ft: mitad del ancho de home (17")
ZONA_ABAJO = 1.62      # ft
ZONA_ARRIBA = 3.21     # ft
ZONA_FUERA = 10

NOMBRES_ZONA = {1: "alto-adentro", 2: "alto-centro", 3: "alto-afuera",
                4: "medio-adentro", 5: "medio-centro", 6: "medio-afuera",
                7: "bajo-adentro", 8: "bajo-centro", 9: "bajo-afuera",
                ZONA_FUERA: "fuera"}


def lado_relativo(df):
    """PlateLocSide con signo del bateador: positivo = adentro. NaN si no se sabe el lado."""
    signo = df["BatterSide"].map({"Right": 1.0, "Left": -1.0})
    return df["PlateLocSide"] * signo


def asignar_zona(altura, lado_rel):
    """Zona 1-9 dentro de la zona de strike (filas alto/medio/bajo, columnas adentro/centro/afuera); 10 fuera."""
    altura = np.asarray(altura, dtype=float)
    lado_rel = np.asarray(lado_rel, dtype=float)
    dentro = (np.abs(lado_rel) <= ZONA_LADO) & (altura >= ZONA_ABAJO) & (altura <= ZONA_ARRIBA)

    tercio_alto = (ZONA_ARRIBA - ZONA_ABAJO) / 3
    fila = np.where(altura > ZONA_ARRIBA - tercio_alto, 0,
                    np.where(altura > ZONA_ABAJO + tercio_alto, 1, 2))          # 0 alto, 1 medio, 2 bajo
    tercio_ancho = 2 * ZONA_LADO / 3
    columna = np.where(lado_rel > ZONA_LADO - tercio_ancho, 0,
                       np.where(lado_rel > -ZONA_LADO + tercio_ancho, 1, 2))   # 0 adentro, 1 centro, 2 afuera

    return np.where(dentro, 1 + 3 * fila + columna, ZONA_FUERA).astype(int)


def preparar_ubicacion(df):
    """Agrega las columnas que usa Location+: lado_relativo, zona, bateador_zurdo, misma_mano, tipo_pitch.

    Quita los pitches sin lado de bateador conocido (no se puede saber qué es adentro).
    """
    df = df[df["BatterSide"].isin(["Right", "Left"])].copy()
    df["lado_relativo"] = lado_relativo(df)
    df["zona"] = asignar_zona(df["PlateLocHeight"], df["lado_relativo"])
    df["bateador_zurdo"] = (df["BatterSide"] == "Left").astype(int)
    df["misma_mano"] = (df["BatterSide"] == df["PitcherThrows"]).astype(int)
    # Categoría con lista fija de tipos: así train, validación y predicción usan los mismos códigos
    df["tipo_pitch"] = pd.Categorical(df["AutoPitchType"].astype(str), categories=TIPOS_VALIDOS)
    return df


def revisar(df):
    """Revisiones de la zona sobre datos limpios."""
    print(f"Pitches con lado de bateador conocido: {len(df):,}")

    hbp = df[df["PitchCall"] == "HitByPitch"]
    print("\n1) Pelotazos (deben estar muy ADENTRO = lado_relativo positivo, para los dos lados):")
    print("   mediana de lado_relativo:", hbp.groupby("BatterSide")["lado_relativo"].median().round(2).to_dict())

    coincide = ((df["zona"] != ZONA_FUERA) == (df["in_strike_zone"] == 1)).mean()
    print(f"\n2) Zonas 1-9 vs in_strike_zone del ISAC: coinciden en {coincide:.3%} de los pitches")

    tomados = df[df["PitchCall"].isin(["StrikeCalled", "BallCalled"])]
    tasa = (tomados.assign(strike=tomados["PitchCall"] == "StrikeCalled")
            .groupby("zona")["strike"].agg(["mean", "size"]))
    tasa.index = [f"{z} {NOMBRES_ZONA[z]}" for z in tasa.index]
    print("\n3) Pitches tomados (sin swing): % cantado strike por zona (dentro debe ser alto, fuera bajo):")
    print((tasa.assign(mean=lambda t: (100 * t["mean"]).round(1))
           .rename(columns={"mean": "% strike", "size": "n"})).to_string())

    reparto = (df["zona"].value_counts(normalize=True).sort_index() * 100).round(1)
    reparto.index = [f"{z} {NOMBRES_ZONA[z]}" for z in reparto.index]
    print("\n4) % de pitches en cada zona:")
    print(reparto.to_string())


if __name__ == "__main__":
    from limpieza import cargar_datos

    revisar(preparar_ubicacion(cargar_datos()))
