# Decisiones y especificación del proyecto · bearsISAC

**Hackathon ISAC 2026 · Stuff+ para la LMB / Diablos Rojos (Harp Helú, 2,240 m)**
Entrega: **2026-10-15** (meta interna 14 oct) · Finalistas 20 oct · Presentaciones 10–11 nov

Este documento es la **fuente de verdad**: si una decisión cambia, se cambia aquí con fecha y motivo.
_Última actualización: 2026-10-05._

## 1. Equipo y roles

| Persona | Perfil | Rol |
|---|---|---|
| Matías | Más programación | Líder técnico: repo, limpieza, features físicas, dashboard, revisión de código |
| Ricardo | Programación media-alta | Líder de modelado: target, modelos, arsenal, scoring, contrafactual de altitud; coordinación |
| Ivana | Algo de programación, fuerte en mates | Valores de carrera, densidad del aire, baseline, validación, efecto de altitud |
| Fran | Menos programación, sabe béisbol | Diccionario, altitud por categoría, revisión con ojo de béisbol, perfil ideal, recomendaciones, editor del documento, presentación |

Reparto sugerido (a confirmar): Matías toma también el modelo de contacto y SHAP; Ricardo y Matías programan en pareja el primer modelo.

## 2. Datos (según `stuff_plus_data_dictionary.csv` de ISAC)

- 1 fila = 1 lanzamiento; 84 columnas; 3 temporadas (`year`).
- El parquet (`stuff_model_df.parquet`) vive una carpeta arriba del repo y **nunca se sube a git**. Salidas y modelos también se guardan fuera del repo (`../salidas/`, `../modelos/`).
- IDs: `PitchUID`, `game_anon_id`, `pitcher_anon_id`, `batter_anon_id`, `catcher_anon_id`. Nunca entran como features.
- No hay estadio en el dataset: solo `altitude_category` ∈ {Extreme Altitude, Medium Altitude, No Altitude}.
- **Reconstrucción de Matías** (`estadios.py`, `altitud.py`): equipos por clustering espectral de pitchers que comparten mitades de juego; local = equipo que lanza en la parte alta; estadio = estadio del local; densidad relativa por juego con k = ay0 / v² de las rectas (el arrastre es proporcional a la densidad; 1.00 = mediana de juegos "No Altitude"); Harp Helú = estadio de altitud extrema con mejor récord de local (Diablos). `altitud.py` empareja por orden los estadios medidos con las altitudes reales de los 20 estadios de la LMB y ajusta una recta densidad medida vs. esperada. Salidas fuera del repo: `../estadios.parquet`, `../altitudes.parquet`.
- Consecuencia: la altitud puede tratarse como variable continua (`densidad_juego`, que además captura temperatura) y Harp Helú puede aislarse. Supuestos a documentar y verificar: identificación de Harp Helú por récord; emparejamiento por orden dentro de cada categoría.
- **Corrección pendiente en `altitud.py`**: la fórmula (1 − 2.25577e-5·h)^5.25588 es la razón de **presión**; la de **densidad** (atmósfera estándar) usa exponente 4.25588 (Harp Helú: 0.80 vs 0.76).
- No hay corredores en base: no se puede construir la RE24 completa (ver paso 5).
- No hay clima (hasta que ISAC diga lo contrario).
- **Pregunta pendiente para ISAC**: rangos/estadios por `altitude_category`, ID anónimo de estadio, clima por partido.

**Regla de oro de features:** una feature es algo que se sabe en el momento en que la bola sale de la mano. Resultado, batazo y ubicación en el plato no entran a Stuff+.

Columnas a **NO** usar como features de Stuff+:

- `ZoneSpeed`, `SpeedDrop`, `ZoneTime`: dependen de la densidad del aire (filtran la altitud de forma escondida).
- `VertApprAngle`, `HorzApprAngle` crudos: dependen de la ubicación. Solo usar ajustados por `PlateLocHeight` (v2).
- `VertRelAngle` crudo: también codifica la ubicación (correlación 0.67–0.84 con `PlateLocHeight` dentro de cada tipo de pitch). Probado el 2026-10-05: sube la mejora sobre el baseline de 1.04 % a 12.35 %, casi toda por ubicación. Sí va en Location+ / Pitching+.
- `Tilt` (repite `SpinAxis`), `VertBreak`, `pfxx`, `pfxz`, coeficientes del polinomio (redundantes).
- `PlateLocHeight`, `PlateLocSide`, `in_strike_zone`: solo Location+ / Pitching+.

## 3. Arquitectura: tres modelos separados

| Modelo | Pregunta | Features | Conteo | Altitud |
|---|---|---|---|---|
| **Stuff+** | ¿Qué tan buena es la bola? | Física + arsenal + altitud + conteo (+ lado del bateador) | Entra; se neutraliza (12 conteos, pesos de la liga) | Entra; se neutraliza (mezcla de la liga) para el score oficial |
| **Location+** | ¿La puso en el lugar correcto? | Ubicación + conteo + lado del bateador + tipo de pitch | Entra; se evalúa en el conteo real, restando el xRV promedio de ese conteo | Opcional |
| **Pitching+** | ¿Qué tan bueno fue este lanzamiento? | Todo | Conteo real, restando el promedio del conteo | Altitud real |

- Pitching+ se entrena desde cero con todas las columnas. No es promedio de Stuff+ y Location+ (el valor de la ubicación depende del pitch: recta arriba buena, sinker arriba malo; y las escalas no se suman). Chequeo: debe correlacionar con los otros dos.
- Terminología para el documento: Stuff+ = física pura (estrella del análisis de altitud); Pitching+ = score final combinado del brief; Location+ = ubicación. Aclarar la ambigüedad del brief.
- Mismo método en los tres: gradient boosting (LightGBM) multiclase → xRV → escala 100 + 10z.

## 4. El pipeline, paso a paso

### Paso 1 · Limpiar (`limpieza.py`, Matías)

- Quitar no-competencia: `PitchCall` ∈ {BallIntentional, AutomaticBall, AutomaticStrike, Undefined}; bunts (`hit_type` contiene "bunt") [pendiente].
- Quitar no describible: `AutoPitchType` = Other; mano desconocida; nulos en física clave.
- Quitar errores: `RelSpeed` ∉ [60, 105]; `Extension` ∉ [3, 8]; |IVB| o |HB| > 30.
- Voltear zurdos: `HorzBreak`, `RelSide`, `HorzRelAngle`, `HorzApprAngle` × (−1); `SpinAxis` → 360 − `SpinAxis` [pendiente los últimos tres]. Después: negativo = hacia el guante siempre.
- Sale: `pitches_clean.parquet` + tabla de filas quitadas por filtro (va al documento).

### Paso 2 · Etiquetar (`objetivo.py`, Ricardo — escrito)

- 5 clases exhaustivas y excluyentes: `ball` 0, `called_strike` 1, `whiff` 2, `foul` 3, `in_play` 4.
- `HitByPitch` = ball. `FoulTip` (si aparece) = whiff. Lo desconocido se quita con aviso.
- Revisar: proporciones (~36 / 16 / 11 / 18 / 17 %); coincidencia 100% con `is_whiff` e `is_swing`; si no coincide, adoptar la definición de ISAC.
- Por qué 5: cada clase vale distinto y tiene causa distinta; "in play" se abre en el paso 7. Chase/rolling/contacto duro no se pierden (Location+ y modelo de contacto).

#### Resultados verificados (corrida del 2026-10-05, 627,733 lanzamientos válidos)

Proporción de cada clase:

| Clase | Lanzamientos | % |
|---|---|---|
| ball | 237,094 | 37.8 % |
| called_strike | 96,841 | 15.4 % |
| whiff | 71,184 | 11.3 % |
| foul | 107,740 | 17.2 % |
| in_play | 114,874 | 18.3 % |

- **`whiff` coincide 100 % con `is_whiff`**: 0 filas fuera de la diagonal (71,184 en `True`, 556,549 en `False`). Nuestra definición de whiff es exactamente la columna `is_whiff`.
- **`is_swing` difiere en 1,119 filas (0.18 % del total)**, el desajuste es **siempre en una sola dirección**: el ISAC marca `is_swing = 1` pero nuestro mapeo las clasifica como **no-swing** (0 filas en la dirección contraria). Son lanzamientos que `PitchCall` registra como *pitch tomado*. Desglose:

  | PitchCall | Filas | Nuestra clase |
  |---|---|---|
  | BallCalled | 457 | ball |
  | StrikeCalled | 390 | called_strike |
  | HitByPitch | 227 | ball |
  | BallinDirt | 45 | ball |

  Son contradicciones entre la columna autoritativa de resultado (`PitchCall`, de Trackman) y el flag auxiliar `is_swing` del ISAC — muy probablemente check-swings que el umpire juzgó como "no hubo swing" (y por eso `PitchCall` = bola/strike cantado) más algún HitByPitch con el bateador en movimiento. **Decisión v0:** el target se arma desde `PitchCall` (el resultado oficial del lanzamiento, que no admite reclasificar un BallCalled a una clase de swing), así que estas 1,119 filas conservan su etiqueta `ball`/`called_strike`. El desacuerdo es solo con `is_swing` (no con el target) y es despreciable; no se descartan. Pendiente de confirmar con Fran/ISAC si alguna categoría de check-swing debiera moverse.

### Paso 3 · Features

- Física v0: `RelSpeed`, `SpinRate`, `InducedVertBreak`, `HorzBreak`, `Extension`, `RelHeight`, `RelSide`.
- Física v1: + `SpinAxis`, `HorzRelAngle`. (`VertRelAngle` descartado para Stuff+: codifica ubicación, ver arriba.)
- Física v2: + VAA ajustado por altura en el plato; + eficiencia de spin (de ax0, az0, SpinRate, velocidad) → H4.
- Contexto: `Balls`, `Strikes` (opción B), `altitude_category`, `BatterSide`.
- Arsenal (por pitcher × temporada): recta primaria = tipo más usado entre Four-Seam/Sinker/Cutter; `velo_diff_vs_fb`, `ivb_diff_vs_fb`, `hb_diff_vs_fb`, `release_diff_vs_fb`, `pitch_usage_pct`.
- `AutoPitchType` no es feature del modelo físico (el modelo ve formas, no etiquetas); sí se usa para arsenal y agregación.

### Paso 4 · Separar datos

- Última temporada = **test sellado**: no se toca hasta la validación final.
- Todo lo demás (hiperparámetros, early stopping, selección de features) se decide con las temporadas anteriores usando GroupKFold por pitcher.
- Motivo: si se eligen hiperparámetros con los mismos datos que se reportan, el resultado sale inflado.

### Paso 5 · Valores de carrera (Ivana)

Sin corredores en base → linear weights con outs. Método que no requiere el orden de los lanzamientos:

- Usar los datos **sin limpiar** (los filtros quitan filas con carreras y resultados). Solo temporadas de entrenamiento.
- Media entrada = (`game_anon_id`, `Inning`, Top/Bottom). Turno = (media entrada, `batter_anon_id`); descartar turnos con más de un lanzamiento final (el bateador volvió a batear en la misma entrada, raro).
- Lanzamiento final del turno: `KorBB` ∈ {Strikeout, Walk} o `PitchCall` ∈ {InPlay, HitByPitch}.
- RE(o), o = 0, 1, 2: como `Outs` nunca baja dentro de una media entrada, las carreras desde que se llegó a o outs = suma de `RunsScored` en lanzamientos con `Outs` ≥ o. RE(o) = promedio sobre las medias entradas que llegaron a o outs. RE(3) = 0.
- Valor del evento en cada lanzamiento final: RE(min(Outs + OutsOnPlay, 3)) − RE(Outs) + RunsScored.
- V(conteo) = promedio del valor del evento final de los turnos que pasaron por ese conteo (pares únicos turno-conteo, porque los fouls con 2 strikes repiten conteo).
- RV_k(b, s), perspectiva del bateador (+ = bueno para el bateador):
  - ball: V(b+1, s) − V(b, s); si b = 3: V_walk − V(3, s).
  - called_strike, whiff: V(b, s+1) − V(b, s); si s = 2: V_K − V(b, 2).
  - foul: igual que strike si s < 2; 0 si s = 2.
  - in_play (v0): promedio del valor del evento de los InPlay en ese conteo − V(b, s). Se reemplaza en el paso 7 por E[valor | contacto, pitch].
  - V_walk, V_K = promedio del valor del evento de las bases por bolas y de los ponches.
- Sanity checks: RE(0) > RE(1) > RE(2); V sube con bolas y baja con strikes; RV_ball > 0, RV_strike < 0; linear weights vs. MLB (HR ≈ +1.4, BB ≈ +0.3, K ≈ −0.27, out en juego ≈ −0.25).
- Aproximaciones a documentar: sin corredores (promedia sobre estados de base); HBP como ball; carreras en pitches no finales (wild pitch, robo) entran en RE pero no se atribuyen al turno.
- Robustez: repetir por `altitude_category` para ver cuánto cambia el ambiente de carreras.

### Paso 6 · Modelo de resultado (`modelo_multiclase.py`, Ricardo — v0 escrito)

- LightGBM `objective="multiclass"`, log-loss (proper scoring rule → probabilidades calibradas).
- Validación: GroupKFold(5) por `pitcher_anon_id`; early stopping con 10% de pitchers apartados dentro del train.
- Predicciones out-of-fold guardadas (cada pitch predicho por un modelo que no vio a su pitcher) → insumo del paso 8.
- **Opción B** para el conteo: `Balls` y `Strikes` entran como features (v0 del código aún no los tiene: pendiente). Motivo: sesgo de selección (sliders se usan en 0-2, rectas en 3-0).
- Baseline justo: frecuencia de cada clase dentro de cada conteo.
- Combinaciones raras (p. ej. curva en 3-0): el modelo combina efecto del conteo y de la física (suma de árboles); conteos raros pesan poco (3-0 ≈ 2%). Vigilar: tabla tipo × conteo (<200 = marcar), revisar predicciones raras, correlación Stuff+ opción A vs B > 0.9. Plan B: pesos de conteo por tipo de pitch.
- Mejora esperada sobre baseline de solo física: pequeña (≈0.5–3%). Es normal: ball/strike depende sobre todo de la ubicación.

Optimización de `min_child_samples` (y demás hiperparámetros):

- Justificación teórica: SE de una proporción = √(p(1−p)/n); con p = 0.11, n = 20 → ±14 pts (95%), n = 200 → ±4 pts. Balance bias/variance.
- Justificación empírica (la que cuenta): probar {20, 50, 100, 200, 500, 1000, 2000} con la misma CV por pitcher en las temporadas de entrenamiento; elegir menor log-loss; si hay empate dentro de la desviación entre partes, elegir el más grande.
- Criterio extra: estabilidad del Stuff+ de cada pitcher entre temporadas.
- Igual para `learning_rate`, `num_leaves`.

### Paso 7 · Modelo de contacto

- Solo lanzamientos in play. Predice tipo de contacto (`hit_type`: rolling, línea, elevado, popup) y/o su valor en carreras, usando también `ExitSpeed`/`Angle` para contacto duro/barrel.
- Valor del contacto por categoría de altitud (en altura los elevados cuestan más → H3).
- Reemplaza el RV fijo de "in play" por E[RV | contacto, pitch].

### Paso 8 · xRV y escala

- xRV = Σ_k P(k | pitch, contexto) · RV_k(conteo), con in_play valuado por el paso 7.
- Stuff+: evaluar cada pitch en los 12 conteos y promediar con pesos de la liga (w_c); idem altitud (mezcla de la liga) y lado del bateador.
- Location+ / Pitching+: conteo real; score = xRV(pitch, conteo real) − xRV promedio de la liga en ese conteo.
- Escala: s = −xRV; Stuff+ = 100 + 10·(s − μ)/σ con μ, σ de toda la liga por temporada (no solo test).
- Agregación: lanzamiento → pitcher × tipo de pitch → pitcher (ponderado por uso).

### Paso 9 · Validación final

- Sobre el test sellado: log-loss/RMSE vs. baseline "xRV promedio por conteo"; curvas de calibración.
- Pitcher holdout (GroupKFold), season holdout (última temporada), park holdout (entrenar sin la categoría de altitud alta y probar en ella).
- Después de validar: reentrenar con todo para los scores finales.

### Paso 10 · Altitud

- La altitud actúa por dos canales: (1) menos movimiento (ya viene en IVB/HB medidos); (2) contacto más caro (lo aprende el modelo vía `altitude_category`).
- Delta de altitud = mismo pitch evaluado en "alta" vs "baja", cambiando la categoría y ajustando el movimiento: IVB_bajo ≈ IVB_alto · ρ_bajo/ρ_alto (idem HB). Factor a validar empíricamente con los mismos pitchers en distintas categorías (efecto fijo por pitcher).
- Densidad por categoría: p = p₀·e^(−h/8400), ρ = p/(R·T). A 2,240 m, ρ ≈ 0.77–0.80 de la del nivel del mar.
- Hipótesis del brief: H1 (spin pierde en altura), H2 (cambio gana relativo), H3 (groundball pitchers valen más), H4 (spin rate solo es insuficiente; usar eficiencia).

### Paso 11 · Productos

- Scores por lanzamiento / pitcher × tipo / pitcher; comparación por altitud.
- Perfil físico ideal para Diablos; recomendaciones operativas (quién gana/pierde en altura, cómo perfilar prospectos).
- Dashboard (Streamlit): simulador de un pitch → Stuff+; lado a lado alta vs baja; mapa de movimiento (IVB vs HB).
- Documento técnico 6–10 págs.; SHAP traducido a lenguaje de coach.

## 5. Convenciones de trabajo

- Rama por tarea (`nombre/tema`), PR con revisión, nunca push a `main`.
- `.gitignore` bloquea `*.parquet`, `*.csv` (excepto `config/*.csv`), `models/`.
- Semilla fija 42. Instalación con `requirements.txt`.
- Notebooks para explorar; lo definitivo va a código `.py`.

## 6. Estado (2026-09-28)

- **Hecho:** `limpieza.py` v0, `modelo.py` (P(whiff | swing), referencia), README, `explorar_columnas.py`, `objetivo.py`, `modelo_multiclase.py` v0 (sin conteo aún).
- **Pendiente inmediato:** inventario del parquet (Matías); pregunta a ISAC sobre altitud/estadios/clima; correr `objetivo.py` y `modelo_multiclase.py` con datos reales; agregar `Balls`/`Strikes`; completar limpieza (bunts, SpinAxis, ángulos); separar test sellado.

## 7. Bitácora de cambios

- **2026-09-24:** arquitectura de 3 modelos; roles; estructura modular.
- **2026-09-24:** diccionario de ISAC revela que no hay estadio ni corredores → altitud por categoría; linear weights con outs.
- **2026-09-28:** opción B para el conteo; Pitching+ desde cero; Location+/Pitching+ en conteo real restando el promedio del conteo; altitud como feature + contexto de evaluación + ajuste de movimiento en el delta; test sellado = última temporada; `min_child_samples` por CV.
- **2026-09-28:** método de RV_k sin depender del orden de los lanzamientos (RE por outs con suma de `RunsScored` en `Outs` ≥ o); usar datos sin limpiar para valores de carrera.
- **2026-10-05:** Matías reconstruye estadios y densidad por juego → altitud continua y Harp Helú identificable.
- **2026-10-05:** `objetivo.py` corrido con datos reales (627,733 lanzamientos). Verificado: proporciones de las 5 clases (ball 37.8 %, called_strike 15.4 %, whiff 11.3 %, foul 17.2 %, in_play 18.3 %); `whiff` coincide 100 % con `is_whiff`; `is_swing` difiere en 1,119 filas (0.18 %), todas pitches tomados (BallCalled 457, StrikeCalled 390, HitByPitch 227, BallinDirt 45) que el ISAC marca como swing — se conservan porque el target se arma desde `PitchCall`.
- **2026-10-05:** `limpieza.py` completo: bunts, volteo de `SpinAxis` (360 − ángulo), `HorzRelAngle` y `HorzApprAngle` para zurdos, tabla de filas quitadas por filtro (9,542 de 635,002 = 1.5 %) y `../pitches_clean.parquet`. En `modelo_multiclase.py` se agrega `SpinAxis` (mejora 1.04 % → 1.04 %, casi nada por sí solo) y se **descarta `VertRelAngle`** para Stuff+: codifica la ubicación (mejora 1.04 % → 12.35 %).
- **2026-10-05:** `run_values.py` (paso 5, Matías). El método solo con outs falla: una base por bolas valía +0.03 carreras (MLB ≈ +0.3) porque el corredor que llega a base no cambia los outs; por eso las bolas en 3-0 salían negativas. **Cambio:** las filas del parquet están en orden cronológico (outs no decrecen en 99.8 % de las medias entradas), así que se cuenta el número de corredores en base (bateadores que terminaron − outs − carreras) y se usa RE(outs, corredores), 12 estados. Resultados con 2024–2025: BB +0.42, 1B +0.54, 2B +0.69, 3B +0.84, HR +1.44, K −0.34, out en juego −0.36; pasan las 7 revisiones (RE baja con outs y sube con corredores, V sube con bolas y baja con strikes, bolas > 0, strikes < 0, pesos suman 1). Limitación: se sabe cuántos corredores hay, no en qué base.
