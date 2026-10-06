# bearsISAC

Hackathon ISAC 2026 · Diablos Rojos · ITAM

Modelo **Stuff+**: califica la calidad de cada pitch a partir de sus propiedades físicas
(velocidad, spin, movimiento, punto de salida), ajustado por la altitud del estadio.
100 = promedio de la liga, 110 = una desviación estándar mejor que el promedio.

**Entrega: 2026-10-15**

## Datos
Los datos del ISAC **no se suben a GitHub**: solo pueden usarse para la competencia.
El archivo `stuff_model_df.parquet` va **una carpeta arriba** del repo:

```
DIABLOS/
├── stuff_model_df.parquet            ← datos del ISAC (no se suben)
├── stuff_plus_data_dictionary.csv    ← diccionario de columnas
├── pitches_clean.parquet             ← lo genera limpieza.py
├── estadios.parquet                  ← lo genera estadios.py
├── altitudes.parquet                 ← lo genera altitud.py
├── salidas/
│   ├── predicciones_oof.parquet      ← lo genera modelo_multiclase.py
│   ├── validacion_multiclase.csv     ← lo genera modelo_multiclase.py
│   ├── folds.csv                     ← lo genera modelo_multiclase.py
│   ├── predicciones_12conteos.parquet← lo genera predecir_12_conteos.py
│   ├── rv_tabla.csv                  ← lo genera run_values.py
│   ├── valores_evento.csv            ← lo genera run_values.py
│   └── pesos_conteo.csv              ← lo genera run_values.py
├── modelos/                          ← lo genera modelo_multiclase.py
│   ├── modelo_resultado.txt          ← modelo final
│   └── modelo_resultado_fold1..5.txt ← un modelo por fold
└── bearsISAC/                        ← este repo
    ├── CLAUDE.md
    ├── limpieza.py
    ├── objetivo.py
    ├── particion.py
    ├── modelo_multiclase.py
    ├── busqueda_min_child.py
    ├── predecir_12_conteos.py
    ├── run_values.py
    ├── estadios.py
    ├── altitud.py
    ├── dictionary.txt
    └── docs/
        ├── guia_proyecto.md
        └── decisions.md
```

Los scripts guardan sus resultados en `DIABLOS/`, fuera del repo, porque también son datos del ISAC.

## Instalación (WSL / Linux)
```bash
sudo apt install python3-venv libgomp1
python3 -m venv ~/venvs/bears
source ~/venvs/bears/bin/activate
pip install pandas pyarrow lightgbm scikit-learn pybaseball
```

Cada vez que abras una terminal nueva, activa el entorno:
```bash
source ~/venvs/bears/bin/activate
```

## Uso
Desde la carpeta del repo, con el entorno activado:
```bash
python limpieza.py              # limpia los datos e imprime la tabla de filas quitadas
python objetivo.py              # revisa el target de 5 resultados contra los datos
python particion.py             # muestra el split entrenamiento / test sellado
python modelo_multiclase.py     # modelo principal: probabilidad de cada resultado
python predecir_12_conteos.py   # predice cada pitch en los 12 conteos (necesita modelo_multiclase.py antes)
python run_values.py            # precio en carreras de cada resultado según el conteo

python busqueda_min_child.py    # (lento, opcional) elige min_child_samples por validación cruzada

python estadios.py    # 1) estadio y densidad del aire de cada juego
python altitud.py     # 2) altitud de cada estadio (necesita estadios.py antes)
```

## Archivos
| Archivo | Qué hace |
|---|---|
| `limpieza.py` | Carga los datos y los limpia: quita pitches no competitivos, bunts, filas vacías y valores imposibles; voltea `HorzBreak`, `RelSide`, `HorzRelAngle`, `HorzApprAngle` y `SpinAxis` de los zurdos. Al correrlo imprime cuántas filas quita cada filtro y guarda `pitches_clean.parquet`. |
| `objetivo.py` | Define el target: traduce `PitchCall` a 5 resultados (`ball`, `called_strike`, `whiff`, `foul`, `in_play`). Al correrlo, revisa las proporciones y que coincida con `is_whiff` e `is_swing` del ISAC. |
| `particion.py` | Separa el **test sellado**: la última temporada (2026) no se usa para entrenar ni para elegir nada, solo para la validación final. `separar_test(df)` devuelve `(entrenamiento, test)`. |
| `modelo_multiclase.py` | Modelo principal. LightGBM multiclase que predice la probabilidad de cada uno de los 5 resultados con la física del pitch (+ `SpinAxis`) y el conteo (`Balls`, `Strikes`). Usa solo el entrenamiento. Validación cruzada de 5 partes por pitcher contra un baseline por conteo, y calibración general y por conteo. Guarda las predicciones fuera de muestra y `folds.csv` en `../salidas/`, y los 5 modelos de fold más el final en `../modelos/`. |
| `busqueda_min_child.py` | Prueba `min_child_samples` ∈ {20 … 2000} con la misma validación cruzada y elige con comparación pareada fold a fold. Resultado: 1000. Tarda bastante (35 entrenamientos). |
| `predecir_12_conteos.py` | Para cada pitch, deja fija la física y predice las 5 probabilidades en los **12 conteos**, con el modelo del fold que no vio a su pitcher. Es el insumo para el Stuff+ neutral al conteo. Guarda `predicciones_12conteos.parquet`. |
| `run_values.py` | Calcula el precio en carreras de cada resultado según el conteo (`RV_k`), con datos sin limpiar de las temporadas de entrenamiento. Guarda `rv_tabla.csv`, `valores_evento.csv` y `pesos_conteo.csv` en `../salidas/`. |
| `estadios.py` | Reconstruye en qué estadio se jugó cada juego (los datos no lo dicen), mide la densidad del aire con el frenado de las rectas e identifica los juegos en el Harp Helú. Genera `estadios.parquet`. |
| `altitud.py` | Calibra la densidad medida con las altitudes reales de los 20 estadios de la LMB y estima la altitud de cada uno. Genera `altitudes.parquet`. |
| `dictionary.txt` | Definiciones de las columnas que usamos. |
| `CLAUDE.md` | Instrucciones para el asistente de IA de cada quien (reglas de datos, cómo trabajar con el equipo). |
| `docs/guia_proyecto.md` | Guía del proyecto: qué construimos, cómo funciona cada pieza y los **contratos entre módulos** (nombres de funciones, archivos y columnas). |
| `docs/decisions.md` | **Fuente de verdad del proyecto:** roles, arquitectura (Stuff+, Location+, Pitching+), el pipeline paso a paso y la bitácora de decisiones. Si una decisión cambia, se cambia ahí con fecha y motivo. |

## Modelo de resultados (5 clases)
Cada lanzamiento termina en uno de 5 resultados, definidos en `objetivo.py` a partir de `PitchCall`:

| Clase | Significado | % de los datos |
|---|---|---|
| `ball` (0) | sin swing, bola (incluye pelotazo) | 37.8 % |
| `called_strike` (1) | sin swing, strike | 15.4 % |
| `whiff` (2) | swing y falló | 11.3 % |
| `foul` (3) | swing y foul | 17.2 % |
| `in_play` (4) | swing y bola en juego | 18.3 % |

`modelo_multiclase.py` predice las 5 probabilidades de cada lanzamiento (suman 1). Más adelante se combinan con el valor en carreras de cada resultado para obtener el xRV y, de ahí, el Stuff+ (ver `docs/decisions.md`, pasos 5–8).

**Resultados actuales** (entrenamiento 2024–2025, validación cruzada por pitcher):
- Mejora sobre el baseline por conteo: **1.00 %** (0.82–1.17 % en los 5 folds). Es poco y es normal: si un pitch termina en bola o strike depende sobre todo de la ubicación, que Stuff+ no ve a propósito. La mejora más grande es en whiff (2.3 %).
- Calibración: la probabilidad predicha y la frecuencia real difieren menos de 1 punto en casi todos los conteos (3-0: 2.2 puntos, con pocos pitches).
- Features: física (`RelSpeed`, `SpinRate`, `InducedVertBreak`, `HorzBreak`, `Extension`, `RelHeight`, `RelSide`, `SpinAxis`) + conteo (`Balls`, `Strikes`). `VertRelAngle` se descartó porque codifica la ubicación.

**Stuff+ neutral al conteo:** `predecir_12_conteos.py` evalúa cada pitch en los 12 conteos. El paso siguiente (`scoring.py`, pendiente) multiplica esas probabilidades por los precios de `run_values.py` y promedia los 12 conteos con `pesos_conteo.csv`.

## Valores de carrera (`run_values.py`)
Una **lista de precios**: cuántas carreras vale cada uno de los 5 resultados según el conteo (12 conteos × 5 resultados = 60 precios). Positivo = bueno para el bateador.

```
xRV del pitch = Σ  P(resultado)  ×  RV(resultado, conteo)
                  (modelo_multiclase)  (run_values)
```

Cómo se calcula:
1. **Corredores en base:** los datos no los traen, pero las filas están en orden cronológico, así que se cuentan: `corredores = bateadores que terminaron − outs − carreras`.
2. **RE(outs, corredores):** carreras promedio desde cada uno de los 12 estados hasta que termina la entrada.
3. **Valor de cada evento:** `RE(después) − RE(antes) + carreras`. Resultado (2024–2025): HR +1.44, 2B +0.69, 1B +0.54, BB +0.42, K −0.34, out en juego −0.36.
4. **V(conteo):** qué tan bien terminan, en promedio, los turnos que pasan por cada conteo.
5. **Precio de cada resultado:** cuánto cambia V. Ejemplo: bola en 2-1 = V(3-1) − V(2-1) = +0.14; whiff en 3-2 = ponche − V(3-2) = −0.43.

Solo usa 2024–2025: la última temporada es el test sellado. Limitación: se sabe cuántos corredores hay, no en qué base.

## Estadios y altitud: cómo funciona
Los datos del ISAC no dicen en qué estadio se jugó cada juego. Lo reconstruimos así:

1. **Equipos:** los pitchers que lanzan en la misma mitad de entrada del mismo juego son compañeros. Agrupándolos salen los 20 equipos de cada temporada.
2. **Estadio:** en la parte alta de la entrada lanza el equipo local, y el local juega en su estadio.
3. **Densidad del aire:** el aire frena la bola. `k = ay0 / v²` en las rectas (Four-Seam) es proporcional a la densidad. 1.00 = estadio sin altitud.
4. **Harp Helú:** entre los estadios de altitud extrema, el del equipo local con mejor récord. Coincide con la tabla pública: los Diablos fueron 71–19 en 2024 y 63–25 en 2025.
5. **Altitud:** se ajusta una recta entre la densidad medida y la densidad esperada por la altitud real de los 20 estadios (R² ≈ 0.96). Con esa recta, la densidad de cada estadio se convierte a metros.

**Limitaciones**
- Solo la identidad del Harp Helú es segura. El resto de estadios se empareja con las altitudes reales por orden dentro de su categoría, sin saber qué estadio anónimo es qué ciudad.
- Los nombres `estadio_XX` pueden cambiar si cambian los datos o la versión de scikit-learn. En otros scripts, usar la columna `harp_helu`, nunca el nombre del estadio.
- La temperatura también cambia la densidad y no viene en los datos: los estadios a nivel del mar tienen los errores más grandes.

**Fuentes de altitudes:** [Mediotiempo, 4 jun 2025](https://www.mediotiempo.com/liga-mexicana-de-beisbol/altitud-sobre-el-nivel-del-mar-estadios-equipos-lmb-mlb) (13 estadios de altura); [Open-Meteo](https://open-meteo.com/) (Estadio Finsus en Huimilpan y estadios a nivel del mar).

## Cómo trabajamos con Git
- Cada quien trabaja en su propia rama: `nombre/tema`.
- Nunca hacer push directo a `main`: usar un pull request.
- Hacer `git pull` antes de empezar a trabajar.
- Nunca subir archivos de datos (`.parquet`, `.csv`).
  
