# bearsISAC

Hackathon ISAC 2026 · Diablos Rojos · ITAM

Tres calificaciones por lanzamiento para la LMB, calibradas por altitud (Diablos Rojos, Harp Helú, ~2,232 m):

- **Stuff+**: ¿qué tan buena es la bola? Solo física (velocidad, spin, movimiento, punto de salida) y arsenal, con el movimiento llevado a nivel del mar para que no dependa del estadio. La versión oficial es la **v3**; la v1 es la base sin altitud.
- **Location+**: ¿la puso donde debía? Solo ubicación, dado el conteo, el bateador y el tipo de pitch.
- **Pitching+**: ¿qué tan bueno fue el lanzamiento completo? Todo junto: física, ubicación, conteo y lado del bateador.

En las tres, 100 = promedio de la liga y 110 = una desviación estándar mejor que el promedio.

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
│   ├── pesos_conteo.csv              ← lo genera run_values.py
│   ├── scores_pitch.parquet          ← Stuff+ por pitch (scoring.py)
│   ├── scores_pitcher_tipo.csv       ← Stuff+ por pitcher × tipo (scoring.py)
│   ├── scores_pitcher.csv            ← Stuff+ por pitcher (scoring.py)
│   ├── predicciones_oof_location.parquet ← lo genera modelo_location.py
│   ├── validacion_location.csv       ← lo genera modelo_location.py
│   ├── scores_location_pitch.parquet ← Location+ por pitch (scoring_location.py)
│   ├── scores_location_pitcher_tipo.csv / scores_location_pitcher.csv
│   ├── xrv_promedio_conteo_location.csv ← promedio que se resta en cada conteo
│   ├── predicciones_oof_pitching.parquet ← lo genera pitching_plus.py
│   ├── validacion_pitching.csv       ← lo genera pitching_plus.py
│   ├── scores_pitching_pitch.parquet ← Pitching+ por pitch (scoring_pitching.py)
│   ├── scores_pitching_pitcher_tipo.csv / scores_pitching_pitcher.csv
│   ├── xrv_promedio_conteo_pitching.csv ← promedio que se resta en cada conteo
│   ├── efecto_altitud.csv            ← efecto medido de la densidad (efecto_altitud.py)
│   ├── precio_in_play_altitud.csv    ← precio de la bola en juego por densidad (precio_in_play_altitud.py)
│   ├── predicciones_oof_stuff_altitud.parquet / predicciones_oof_stuff_arsenal.parquet ← Stuff+ v2 y v3
│   ├── park_holdout.csv              ← park_holdout.py
│   ├── scores_stuff_altitud_pitch.parquet ← Stuff+ v3 en nivel del mar y Harp Helú por pitch (contrafactual_altitud.py)
│   ├── delta_altitud_tipo.csv / delta_altitud_pitcher_tipo.csv / delta_altitud_pitcher.csv
│   └── calibracion_stuff_altitud.json ← media, sd y precio de bola en juego
├── modelos/
│   ├── modelo_resultado.txt          ← Stuff+: modelo final (modelo_multiclase.py)
│   ├── modelo_resultado_fold1..5.txt ← Stuff+: un modelo por fold
│   ├── modelo_location.txt           ← Location+: modelo final (modelo_location.py)
│   ├── modelo_location_fold1..5.txt  ← Location+: un modelo por fold
│   ├── modelo_pitching.txt           ← Pitching+: modelo final (pitching_plus.py)
│   ├── modelo_pitching_fold1..5.txt  ← Pitching+: un modelo por fold
│   └── modelo_stuff_altitud*.txt, modelo_stuff_arsenal*.txt ← Stuff+ v2 y v3 (final y un modelo por fold)
└── bearsISAC/                        ← este repo
    ├── CLAUDE.md
    ├── requirements.txt
    ├── limpieza.py
    ├── objetivo.py
    ├── particion.py
    ├── run_values.py
    ├── modelo_multiclase.py
    ├── busqueda_min_child.py
    ├── predecir_12_conteos.py
    ├── scoring.py
    ├── zona.py
    ├── modelo_location.py
    ├── busqueda_min_child_location.py
    ├── scoring_location.py
    ├── pitching_plus.py
    ├── scoring_pitching.py
    ├── efecto_altitud.py
    ├── altitud_movimiento.py
    ├── precio_in_play_altitud.py
    ├── arsenal.py
    ├── modelo_stuff_altitud.py
    ├── modelo_stuff_arsenal.py
    ├── park_holdout.py
    ├── contrafactual_altitud.py
    ├── estadios.py
    ├── altitud.py
    └── docs/
        ├── guia_proyecto.md
        ├── decisions.md
        └── fisica_altitud.md
```

Los scripts guardan sus resultados en `DIABLOS/`, fuera del repo, porque también son datos del ISAC.

## Instalación (WSL / Linux)
```bash
sudo apt install python3-venv libgomp1
python3 -m venv ~/venvs/bears
source ~/venvs/bears/bin/activate
pip install -r requirements.txt
```

En Windows (PowerShell): `python -m venv .venv`, `.venv\Scripts\Activate.ps1` y `pip install -r requirements.txt`.

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
python run_values.py            # precio en carreras de cada resultado según el conteo

# Stuff+
python modelo_multiclase.py     # modelo de física: probabilidad de cada resultado (escribe folds.csv)
python predecir_12_conteos.py   # predice cada pitch en los 12 conteos (necesita modelo_multiclase.py)
python scoring.py               # Stuff+ (necesita predecir_12_conteos.py y run_values.py)

# Location+ (necesita folds.csv de modelo_multiclase.py y rv_tabla.csv de run_values.py)
python zona.py                  # revisiones de la zona de strike y del lado del bateador
python modelo_location.py       # modelo de ubicación: probabilidad de cada resultado
python scoring_location.py      # Location+ (necesita modelo_location.py)

# Pitching+ (necesita folds.csv, rv_tabla.csv, y scores_pitch.parquet y scores_location_pitch.parquet
# para la revisión de correlaciones: corre antes scoring.py y scoring_location.py)
python pitching_plus.py         # modelo que ve todo: física + ubicación + conteo + lado del bateador
python scoring_pitching.py      # Pitching+ (necesita pitching_plus.py)

# (lentos, opcionales) eligen min_child_samples por validación cruzada
python busqueda_min_child.py            # Stuff+    (resultado: 1000)
python busqueda_min_child_location.py   # Location+ (resultado: 2000)

python estadios.py    # 1) estadio y densidad del aire de cada juego
python altitud.py     # 2) altitud de cada estadio (necesita estadios.py antes)

# Altitud y Stuff+ v3 (necesitan estadios.py, folds.csv de modelo_multiclase.py y rv_tabla.csv de run_values.py; en este orden)
python efecto_altitud.py           # 1) cuánto cambian IVB, HB, spin y velocidad con la densidad (mismo pitcher y tipo)
python precio_in_play_altitud.py   # 2) cuánto más vale un batazo en juego con menos aire
python modelo_stuff_altitud.py     # 3) Stuff+ v2: movimiento a nivel del mar + densidad
python modelo_stuff_arsenal.py     # 4) Stuff+ v3 (oficial): v2 + arsenal
python contrafactual_altitud.py    # 5) el mismo pitch en nivel del mar y en Harp Helú, con el cambio por canal
python contratacion.py             # 6) ranking de pitchers para Harp Helú + prueba retrospectiva 2024 → 2025 (necesita scoring_pitching.py y scoring_location.py)
python park_holdout.py             # (opcional) entrenar sin Harp Helú / sin altura extrema y probar ahí
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
| `scoring.py` | Convierte probabilidades en carreras y en escala 100 + 10z. `calcular_xrv(df, "neutral")` promedia los 12 conteos (Stuff+); `calcular_xrv(df, "real")` usa el conteo real y le resta el promedio de ese conteo (Location+, Pitching+). Al correrlo genera el Stuff+ y sus revisiones. |
| `zona.py` | Ubicación vista desde el bateador: `lado_relativo` (+ = adentro, para derechos y zurdos) y zonas 1–9 dentro de la zona de strike + 10 fuera. `preparar_ubicacion(df)` agrega las columnas que usa Location+. |
| `modelo_location.py` | Modelo de Location+: LightGBM multiclase con ubicación, conteo, mano del bateador y tipo de pitch (sin física). Usa **los mismos folds** que Stuff+ (`folds.csv`). Guarda predicciones fuera de muestra y modelos. |
| `busqueda_min_child_location.py` | Igual que `busqueda_min_child.py`, para Location+. Resultado: 2000. |
| `scoring_location.py` | Location+ por pitch, pitcher × tipo y pitcher, más 5 revisiones (zona por conteo, tasa de bolas, correlación con Stuff+, estabilidad entre temporadas). |
| `pitching_plus.py` | Modelo de Pitching+: LightGBM multiclase con **todo**: las 8 columnas de física de Stuff+, la ubicación de Location+, conteo y lado del bateador (sin `tipo_pitch` ni altitud). Usa **los mismos folds** que Stuff+ y Location+ (`folds.csv`). Guarda predicciones fuera de muestra y modelos. |
| `scoring_pitching.py` | Pitching+ por pitch, pitcher × tipo y pitcher (`calcular_xrv(df, "real")` y escala 100 + 10z), más 5 revisiones, incluida la correlación con Stuff+ y Location+. |
| `efecto_altitud.py` | Mide con datos de la LMB cuánto cambian IVB, HB, spin y velocidad con la densidad, **dentro del mismo pitcher y tipo de pitch** (así el equipo no se confunde con el estadio). `agregar_densidad(df)` une `densidad_juego` a los pitches. |
| `altitud_movimiento.py` | `ajustar_movimiento(ivb, hb, rho_origen, rho_destino, tipo)` con el efecto medido (aditivo por tipo) y `normalizar_movimiento(df)`, que lleva IVB y HB a nivel del mar (`IVB_n`, `HB_n`). |
| `precio_in_play_altitud.py` | Cuánto más vale un batazo en juego cuando el aire es menos denso, comparando al mismo bateador. |
| `arsenal.py` | Features de arsenal: diferencias de velocidad, movimiento y release contra la recta primaria del pitcher, y % de uso del tipo. |
| `modelo_stuff_altitud.py` | Stuff+ v2: movimiento normalizado + `densidad_juego`. Mismos folds que Stuff+ v1. |
| `modelo_stuff_arsenal.py` | **Stuff+ v3 (oficial):** v2 + arsenal. Compara con v2 fold a fold. |
| `park_holdout.py` | Entrena sin Harp Helú (o sin los estadios de altura extrema) y prueba ahí, con v1, v2 sin densidad y v2. |
| `contrafactual_altitud.py` | Califica cada pitch en el aire de nivel del mar y en el de Harp Helú y separa el cambio en tres canales (movimiento, resultado, precio de la bola en juego). |
| `contratacion.py` | Ranking de pitchers para Harp Helú: rating = ejecución (Pitching+ ajustado por poca muestra) + ajuste de altitud, en carreras por 100 pitches; marca a los pitchers de los Diablos; prueba retrospectiva del ranking de 2024 contra 2025. |
| `run_values.py` | Calcula el precio en carreras de cada resultado según el conteo (`RV_k`), con datos sin limpiar de las temporadas de entrenamiento. Guarda `rv_tabla.csv`, `valores_evento.csv` y `pesos_conteo.csv` en `../salidas/`. |
| `estadios.py` | Reconstruye en qué estadio se jugó cada juego (los datos no lo dicen), mide la densidad del aire con el frenado de las rectas e identifica los juegos en el Harp Helú. Genera `estadios.parquet`. |
| `altitud.py` | Calibra la densidad medida con las altitudes reales de los 20 estadios de la LMB y estima la altitud de cada uno. Genera `altitudes.parquet`. |
| `CLAUDE.md` | Instrucciones para el asistente de IA de cada quien (reglas de datos, cómo trabajar con el equipo). |
| `docs/guia_proyecto.md` | Guía del proyecto: qué construimos, cómo funciona cada pieza y los **contratos entre módulos** (nombres de funciones, archivos y columnas). |
| `docs/fisica_altitud.md` | Física de la altitud: densidad del aire vs. altitud, Magnus, arrastre y reescalado del movimiento, con sus supuestos. |
| `requirements.txt` | Librerías de Python del proyecto. |
| `docs/decisions.md` | **Fuente de verdad del proyecto:** roles, arquitectura (Stuff+, Location+, Pitching+), el pipeline paso a paso y la bitácora de decisiones. Si una decisión cambia, se cambia ahí con fecha y motivo. |

## Modelo de resultados (5 clases)
Cada lanzamiento termina en uno de 5 resultados, definidos en `objetivo.py` a partir de `PitchCall`:

| Clase | Significado | % de los datos |
|---|---|---|
| `ball` (0) | sin swing, bola (incluye pelotazo) | 37.9 % |
| `called_strike` (1) | sin swing, strike | 15.5 % |
| `whiff` (2) | swing y falló | 11.4 % |
| `foul` (3) | swing y foul | 17.2 % |
| `in_play` (4) | swing y bola en juego | 18.0 % |

`modelo_multiclase.py` predice las 5 probabilidades de cada lanzamiento (suman 1). Más adelante se combinan con el valor en carreras de cada resultado para obtener el xRV y, de ahí, el Stuff+ (ver `docs/decisions.md`, pasos 5–8).

**Resultados actuales** (entrenamiento 2024–2025, validación cruzada por pitcher):
- Mejora sobre el baseline por conteo: **1.00 %** (0.82–1.17 % en los 5 folds). Es poco y es normal: si un pitch termina en bola o strike depende sobre todo de la ubicación, que Stuff+ no ve a propósito. La mejora más grande es en whiff (2.3 %).
- Calibración: la probabilidad predicha y la frecuencia real difieren menos de 1 punto en casi todos los conteos (3-0: 2.2 puntos, con pocos pitches).
- Features: física (`RelSpeed`, `SpinRate`, `InducedVertBreak`, `HorzBreak`, `Extension`, `RelHeight`, `RelSide`, `SpinAxis`) + conteo (`Balls`, `Strikes`). `VertRelAngle` se descartó porque codifica la ubicación.

**Stuff+ neutral al conteo:** `predecir_12_conteos.py` evalúa cada pitch en los 12 conteos y `scoring.py` multiplica esas probabilidades por los precios de `run_values.py` y promedia los 12 conteos con `pesos_conteo.csv`. Así, un slider no sale mejor solo porque se usa más en 0-2. Resultado v1 (sin altitud): a nivel pitcher, Stuff+ vs tasa de whiff r = 0.43; arriba quedan rectas de 92–97 mph y rompientes de spin alto (Stuff+ 117–124), abajo sinkers de poco spin (81–84).

## Location+ (`zona.py`, `modelo_location.py`, `scoring_location.py`)
Mide **la ubicación**, no la física. El modelo ve dónde cruzó la bola (`PlateLocHeight`, `lado_relativo`), el conteo, la mano del bateador (`bateador_zurdo`, `misma_mano`) y el tipo de pitch, y predice las mismas 5 probabilidades.

```
xRV         = Σ  P(resultado | ubicación, conteo real, ...)  ×  RV(resultado, conteo real)
Location+   = 100 + 10 · z( −(xRV − xRV promedio de ese conteo) )      por temporada
```

- **Por qué el conteo real:** dónde conviene tirar depende del conteo (en 3-0 hay que meterla, en 0-2 conviene que persiga afuera).
- **Por qué restar el promedio del conteo:** estar en 0-2 ya es ventaja del pitcher; eso no es mérito de la ubicación.
- **Lado relativo:** `PlateLocSide` se mide desde el catcher; se le cambia el signo para los bateadores zurdos para que **+ siempre sea adentro**. Comprobado con los pelotazos: quedan en +2.0 ft para los dos lados.
- **Mismos folds que Stuff+:** se leen de `folds.csv`, así cada pitch se predice en los dos modelos sin que ninguno haya visto a su pitcher, y sus scores se pueden comparar.

**Resultados actuales** (entrenamiento 2024–2025, validación cruzada por pitcher, `min_child_samples = 2000`):
- Mejora sobre el baseline por conteo: **29.2 %** (Stuff+: 1.0 %). Lo más fuerte es ball (52.7 %) y called_strike (24.3 %): la ubicación explica bola/strike y la física explica el whiff.
- Calibración: menos de 1 punto de diferencia en todos los conteos y menos de 0.3 en todas las zonas.
- En 3-0, dentro de la zona ≈ 114 y fuera 89; en 0-2, el centro de la zona es lo peor (97) y las esquinas bajas lo mejor (101–102).
- A nivel pitcher: Location+ vs tasa de bolas r = −0.74; Location+ vs Stuff+ v3 r = −0.03 (miden cosas distintas). Estabilidad de 2024 a 2025: Location+ r = 0.62, Stuff+ v3 r = 0.78.

## Pitching+ (`pitching_plus.py`, `scoring_pitching.py`)
Mide **el lanzamiento completo**. Es un **tercer modelo entrenado desde cero**, no un promedio de Stuff+ y Location+: el valor de la ubicación depende del pitch (una recta con mucho IVB arriba de la zona es buena; un sinker arriba es malo), y un promedio no puede capturar esa interacción. Además, las escalas de los dos no se pueden sumar. El modelo ve las dos cosas a la vez y descubre esas interacciones solo.

```
Features   = física (8, como Stuff+) + PlateLocHeight, lado_relativo (como Location+)
             + Balls, Strikes + bateador_zurdo, misma_mano      (sin tipo_pitch, sin altitud todavía)
Pitching+  = 100 + 10 · z( −(xRV − xRV promedio de ese conteo) )      por temporada, conteo real
```

- **Sin `tipo_pitch`:** la física ya codifica la forma del pitch, igual que en Stuff+.
- **Misma partición que Stuff+ y Location+** (`folds.csv`): cada pitch se predice en los tres modelos sin que ninguno haya visto a su pitcher, así los scores son comparables pitch a pitch.
- **Altitud:** `FEATURES_PITCHING` es una lista pensada para agregar `densidad_juego` en una v2.

**Resultados actuales** (entrenamiento 2024–2025, validación cruzada por pitcher, `min_child_samples = 1000`; corridos por Ricardo, ver `docs/decisions.md`):
- Mejora sobre el baseline por conteo: **30.3 %** (Stuff+: 1.0 %, Location+: 29.2 %). Casi todo viene de la ubicación: ball 53.6 % (en Stuff+ es 0.02 %). Calibración general ≤ 0.04 puntos; por conteo, máximo 1.13 (en 3-0).
- Pitching+ es la ejecución total, **dominada por la ubicación**: correlaciona con Location+ (r = 0.87) y menos con Stuff+ v3 (r = 0.28), mientras que Stuff+ y Location+ casi no se relacionan entre sí (r = 0.04).
- A nivel pitcher: Pitching+ vs tasa de bolas r = −0.73 y vs tasa de whiff r = +0.27: combina los dos canales. Estabilidad de 2024 a 2025: r = 0.60 (Stuff+: 0.78, Location+: 0.62).
- No es "mejor" que Stuff+: responde otra pregunta. Stuff+ mide la calidad intrínseca de la bola; Pitching+, qué tan bien salió ese lanzamiento.

| Score | Pregunta | Ve | Mejora sobre baseline | Estabilidad 2024→2025 |
|---|---|---|---|---|
| Stuff+ | ¿Qué tan buena es la bola? | física + conteo | 1.0 % | 0.78 |
| Location+ | ¿La puso donde debía? | ubicación + conteo + lado + tipo de pitch | 29.2 % | 0.62 |
| Pitching+ | ¿Qué tan bueno fue el lanzamiento? | todo | 30.3 % | 0.60 |

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

## Altitud en los scores (Stuff+ v3 y contrafactual)
Meter la densidad como una columna más de LightGBM no basta: hay ~20 estadios, Harp Helú es la casa de los Diablos (equipo y estadio se mezclan) y el movimiento medido ya viene reducido por el aire. Por eso la altitud entra por **tres canales**, cada uno medido por su lado:

1. **Movimiento.** `efecto_altitud.py` compara al **mismo pitcher con el mismo tipo de pitch** en aires distintos (482 mil pitches, 766 pitchers). En Harp Helú la recta pierde **3.7 pulgadas de IVB**; el spin no cambia. El movimiento cae **más que proporcionalmente** a la densidad (elasticidad ≈ 1.3, no 1.0 como dice la física simple) y el cambio crece con cuánto se mueve el pitch, así que usamos el efecto medido (`dx/dρ = g0 + κ·x`, una fórmula exacta e invertible; `altitud_movimiento.py`). `modelo_stuff_*` entrena con el movimiento llevado a nivel del mar, para que Stuff+ califique al pitcher y no al estadio (en v1, los pitches de Harp Helú salían con 97 vs. 100 en el resto; ahora 99.3 vs. 100.1).
2. **Resultado.** `densidad_juego` entra al modelo para que aprenda cómo cambian whiff, foul y bola en juego con el aire. **Es el canal menos validado:** en el park holdout no mejoró de forma consistente. Ojo: el modelo se entrena con movimiento a nivel del mar más `densidad_juego`, así que en el escenario Harp Helú la densidad ya trae el costo del movimiento perdido y **no se baja el movimiento otra vez** (hacerlo contaba la pérdida dos veces; auditoría 2026-10-21).
3. **Precio de la bola en juego.** Un batazo en juego cuesta **+0.062 carreras más** en Harp Helú que a nivel del mar (mismo bateador, `precio_in_play_altitud.py`).

**Stuff+ v3 (oficial)** agrega arsenal (diferencias contra la recta primaria, que con el movimiento normalizado no dependen de la altitud): mejora 1.25 % sobre el baseline por conteo (v1: 1.00 %), y gana a v2 en los 5 folds.

**Park holdout** (entrenar sin Harp Helú y probar ahí): el método generaliza, con mejora ≈ 1 % también en estadios que el modelo no vio. Lo robusto es el movimiento normalizado; la densidad como feature es inconclusa.

**Resultado del contrafactual** (el mismo pitch, nivel del mar → Harp Helú; puntos de Stuff+):

| Tipo | Movimiento | Resultado | Bola en juego | **Total** | **Relativo al promedio** |
|---|---|---|---|---|---|
| Four-Seam | −6.9 | −0.2 | −17.2 | −24.3 | **−2.4** |
| Sinker | −2.9 | −2.5 | −21.0 | −26.4 | −4.5 |
| Changeup | −1.1 | −1.8 | −18.3 | −21.1 | +0.8 |
| Slider | −1.4 | −0.1 | −16.7 | −18.1 | +3.8 |
| Curveball | −1.7 | +0.9 | −14.6 | −15.3 | +6.6 |

Un pitch promedio pierde ≈ 22 puntos en Harp Helú, casi todo por el precio de la bola en juego (un parque con mucha carrera, efecto casi igual para todos). Lo que distingue a un tipo de otro es el cambio **relativo**: las rectas pierden más que el promedio (sobre todo por movimiento) y los rompientes pierden menos.

**Límites:** un solo efecto por tipo de pitch; un solo precio para todos los batazos (sin modelo de contacto, así que no se prueba que los pitchers de rolling pierdan menos); el efecto del bateo compara al mismo bateador pero no controla clima ni pitcher; la eficiencia de spin (H4) no se modeló. Detalle y números en `docs/decisions.md` (Paso 10).

**A quién contratar (`contratacion.py`).** Rating = ejecución (Pitching+, encogido hacia un promedio previo cuando hay pocos pitches; la confiabilidad se mide con mitades de juegos) + ajuste por altura, en carreras por 100 pitches contra un pitcher promedio. Se probó con datos reales: el ranking hecho con 2024 acertó en 2025 de forma **moderada** (r = 0.38 con K−BB%, 0.24 con carreras por turno; los 20 primeros superaron al promedio y a los 20 últimos), así que sirve para una **lista corta**, no para garantizar un nombre. Hallazgo: Stuff+ solo casi no predice el resultado del año siguiente; la base del ranking es Pitching+. El ajuste por altura sale del modelo y no se pudo validar contra resultados reales.

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
- Nunca subir archivos de datos (`.parquet`, `.csv`). El `.gitignore` los bloquea (salvo `config/*.csv`), igual que `modelos/`, `salidas/` y `graphify-out/`; aun así, revisar `git status` antes de cada commit.
  

## Dashboard (Streamlit)
Tablero de decisión para los Diablos Rojos, pensado para usarse sin leer código: navegación lateral, texto de ayuda en cada número (signo de interrogación) y una nota "cómo leerlo" en cada pantalla. Usa `dashboard.py` y `simulador_altitud.py` y necesita las salidas de `../salidas/` y `../modelos/`:

```
streamlit run dashboard.py
```

| Sección | Qué responde |
|---|---|
| Resumen | La pregunta, cuatro números clave, lo que encontramos y la lista corta recomendada |
| Candidatos | A quién contratar: filtros (muestra mínima, pitch principal, Diablos), gráfica de rating con su margen de incertidumbre, tabla, perfil de cada pitcher y descarga en CSV |
| Laboratorio de pitches | El mismo pitch a nivel del mar y en Harp Helú (mano, aire de la medición, recta principal), con la pérdida separada por canal y el mapa de movimiento |
| Efecto de la altura | Qué tipos de pitch pierden más o menos que el promedio, con el detalle por canal |
| Metodología y glosario | Definiciones, cómo se construye, calidad de los modelos, validación del ranking y límites |

Los scores de pitcher (Stuff+, Location+, Pitching+) se muestran en escala **entre pitchers** (100 = pitcher promedio, 110 = una desviación estándar mejor); los scores por pitch suelto usan otra escala (ver `docs/decisions.md`, Paso 8).
