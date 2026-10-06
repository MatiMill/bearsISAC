# Decisiones y especificación del proyecto · bearsISAC

**Hackathon ISAC 2026 · Stuff+ para la LMB / Diablos Rojos (Harp Helú, 2,232 m)**
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
- **Corregido en `altitud.py` (2026-10-05)**: se usa el exponente de **densidad** 4.25588 (antes 5.25588, que es el de presión). Recta: medida = 0.957·esperada + 0.058, R² = 0.962; Harp Helú estimado 2,260 m (real 2,232 m).
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

#### Resultados verificados (re-corrida 2026-10-05 con limpieza completa, 625,460 lanzamientos válidos)

Proporción de cada clase (tras quitar bunts y filas sin `SpinAxis`; solo cambió `in_play`, que incluía los bunts):

| Clase | Lanzamientos | % |
|---|---|---|
| ball | 237,094 | 37.9 % |
| called_strike | 96,841 | 15.5 % |
| whiff | 71,184 | 11.4 % |
| foul | 107,740 | 17.2 % |
| in_play | 112,601 | 18.0 % |

- **`whiff` coincide 100 % con `is_whiff`**: 0 filas fuera de la diagonal (71,184 en `True`, 554,276 en `False`). Nuestra definición de whiff es exactamente la columna `is_whiff`.
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

### Paso 4 · Separar datos (`particion.py`, Ricardo — escrito)

- Última temporada = **test sellado**: no se toca hasta la validación final.
- Todo lo demás (hiperparámetros, early stopping, selección de features) se decide con las temporadas anteriores usando GroupKFold por pitcher.
- Motivo: si se eligen hiperparámetros con los mismos datos que se reportan, el resultado sale inflado.
- `separar_test(df) -> (entrenamiento, test)`: test = `df[year == year.max()]`, entrenamiento = el resto. Genérica (cualquier `df` con columna `year`); el Paso 6 la importa con `from particion import separar_test`.

#### Resultados verificados (re-corrida 2026-10-05, limpieza completa, 625,460 lanzamientos)

Test sellado = temporada **2026** (la última; parcial, por eso no es un tercio exacto).

| Parte | Lanzamientos | % | Pitchers distintos |
|---|---|---|---|
| entrenamiento (2024+2025) | 494,804 | 79.1 % | 829 |
| test (2026) | 130,656 | 20.9 % | 549 |

- **Pitchers del test:** 301 (54.8 %) también aparecen en entrenamiento; 248 (45.2 %) son **nuevos** (solo en test).
- Consecuencia: la validación final (Paso 9) mezcla dos efectos a la vez —temporada nueva y pitchers nuevos—. Para un *season holdout* puro conviene reportar métricas separando los 301 pitchers vistos de los 248 nuevos.

### Paso 5 · Valores de carrera (`run_values.py`, Matías — hecho)

Linear weights con **estado (outs, corredores)**. El dataset no dice qué corredores hay, pero las filas vienen en **orden cronológico**, así que se puede contar CUÁNTOS: `corredores = bateadores que terminaron su turno − outs − carreras` (12 estados = 3 niveles de outs × 4 de corredores, 0–3).

> **Corrección 2026-10-05 (supera el plan "solo con outs"):** medir RE solo por outs fallaba — una base por bolas valía +0.03 carreras (MLB ≈ +0.3), porque el corredor que llega a base no cambia los outs, y por eso las bolas en 3-0 salían negativas. Reconstruir el número de corredores arregla eso (los outs no decrecen dentro de la media entrada en el 99.8 %, así que el conteo es confiable).

- Usar los datos **sin limpiar** (los filtros quitan filas con carreras y resultados). Solo temporadas de entrenamiento (2026 sellado).
- Media entrada = (`game_anon_id`, `Inning`, Top/Bottom). Turno = (media entrada, `batter_anon_id`); descartar turnos con más de un lanzamiento final.
- Lanzamiento final del turno: `KorBB` ∈ {Strikeout, Walk} o `PitchCall` ∈ {InPlay, HitByPitch}.
- **RE(outs, corredores):** carreras promedio desde ese estado hasta que termina la media entrada. RE = 0 al terminar la entrada.
- Valor de cada final de turno = RE(estado al terminar) − RE(estado al empezar) + `RunsScored` del turno.
- V(conteo) = promedio del valor final de los turnos que pasaron por ese conteo (pares únicos turno-conteo, porque los fouls con 2 strikes repiten conteo).
- RV_k(b, s), perspectiva del bateador (+ = bueno para el bateador):
  - ball: V(b+1, s) − V(b, s); si b = 3: V_walk − V(3, s).
  - called_strike, whiff: V(b, s+1) − V(b, s); si s = 2: V_K − V(b, 2).
  - foul: igual que strike si s < 2; 0 si s = 2.
  - in_play (v0): promedio del valor de los InPlay en ese conteo − V(b, s). Se reemplaza en el paso 7 por E[valor | contacto, pitch].
- **Resultados verificados (2024–2025):** linear weights BB +0.42, 1B +0.54, 2B +0.69, 3B +0.84, HR +1.44, K −0.34, out en juego −0.36 (cerca de MLB). Pasan las 7 revisiones: RE baja con outs y sube con corredores, V sube con bolas y baja con strikes, toda bola > 0, todo strike < 0, los pesos suman 1.
- Salidas (contrato §6): `../salidas/rv_tabla.csv` (`balls, strikes, outcome_class, rv`), `valores_evento.csv`, `pesos_conteo.csv`.
- Aproximaciones: se sabe **cuántos** corredores hay, no en qué base; HBP como ball; carreras en pitches no finales (wild pitch, robo) entran en RE pero no se atribuyen al turno.
- Robustez: repetir por `altitude_category` para ver cuánto cambia el ambiente de carreras.

### Paso 6 · Modelo de resultado (`modelo_multiclase.py`, Ricardo — v1 escrito, opción B, sin altitud)

- LightGBM `objective="multiclass"`, log-loss (proper scoring rule → probabilidades calibradas).
- Validación: GroupKFold(5) por `pitcher_anon_id`; early stopping con 10% de pitchers apartados dentro del train.
- Predicciones out-of-fold guardadas (cada pitch predicho por un modelo que no vio a su pitcher) → insumo del paso 8.
- **Opción B** para el conteo: `Balls` y `Strikes` entran como features (v1 ya los incluye). Motivo: sesgo de selección (sliders se usan en 0-2, rectas en 3-0).
- Baseline justo: frecuencia de cada clase dentro de cada conteo.
- Combinaciones raras (p. ej. curva en 3-0): el modelo combina efecto del conteo y de la física (suma de árboles); conteos raros pesan poco (3-0 ≈ 2%). Vigilar: tabla tipo × conteo (<200 = marcar), revisar predicciones raras, correlación Stuff+ opción A vs B > 0.9. Plan B: pesos de conteo por tipo de pitch.
- Mejora esperada sobre baseline de solo física: pequeña (≈0.5–3%). Es normal: ball/strike depende sobre todo de la ubicación.

#### Resultados verificados (v1 sin altitud, re-corrida 2026-10-05 con limpieza completa, entrenamiento 494,804 / 829 pitchers)

- Config: LightGBM multiclase con `min_child_samples=1000` (elegido por CV, ver abajo). Features: **8 físicas (incl. `SpinAxis`, física v1)** + `Balls`, `Strikes`. Test 2026 sellado (no entra al modelo ni al OOF). Código listo para altitud: v2 = añadir `"densidad_juego"` a `FEATURES` y nada más.
- **Mejora sobre el baseline por conteo: 1.00 %** de log-loss (0.82–1.17 % entre los 5 folds; log-loss modelo ≈ 1.431 vs baseline ≈ 1.446). Es lo esperado: el baseline ya conoce el conteo y la física no ve la ubicación. `SpinAxis` por sí solo casi no mueve la aguja (sigue en ~1.0 %), pero es física legítima y puede ayudar al interactuar con la altitud en v2.
- **La señal de la física está casi toda en whiff.** Log-loss binaria (clase vs. resto), mejora del modelo sobre el baseline por conteo: whiff **2.30 %**, in_play 0.80 %, foul 0.71 %, called_strike 0.52 %, ball **0.02 %**. Confirma la hipótesis: la física separa swings fallados y calidad de contacto; para ball/strike cantado no aporta (eso es ubicación/conteo).
- **Calibración general**: predicha vs real ≤0.2 pp por clase. **Por conteo**: diferencias de ~1 pp o menos salvo en **3-0** (el conteo más raro, n≈6,600, ~1.3 % del total), donde el modelo sobre-predice swings un poco porque arrastra probabilidad desde conteos vecinos con tan pocos datos. Vigilado, no es bug; el Plan B (pesos de conteo por tipo de pitch) lo atacaría si molesta.
- Salidas fuera del repo: `../salidas/predicciones_oof.parquet` (contrato: `PitchUID, pitcher_anon_id, Balls, Strikes, outcome_class, p_ball, p_called_strike, p_whiff, p_foul, p_in_play`), `../salidas/validacion_multiclase.csv`, `../salidas/folds.csv`, `../modelos/modelo_resultado_fold{1..5}.txt` + `modelo_resultado.txt`.

Optimización de `min_child_samples` (y demás hiperparámetros):

- Justificación teórica: SE de una proporción = √(p(1−p)/n); con p = 0.11, n = 20 → ±14 pts (95%), n = 200 → ±4 pts. Balance bias/variance.
- Justificación empírica (la que cuenta): probar {20, 50, 100, 200, 500, 1000, 2000} con la misma CV por pitcher en las temporadas de entrenamiento.
- **Regla de elección (corregida 2026-10-05):** elegir el de menor log-loss; si la diferencia **pareada fold a fold** con otro valor más grande no es consistente (ese valor menor no gana en al menos 4 de 5 folds), preferir el más grande. _No usar la desviación de la log-loss entre folds como tolerancia de empate: está dominada por la distinta dificultad de cada fold, que afecta por igual a todos los valores; la comparación pareada (misma partición) cancela ese término común y es ~18× más fina._
- Criterio extra: estabilidad del Stuff+ de cada pitcher entre temporadas.
- Igual para `learning_rate`, `num_leaves`.

#### Resultado de la búsqueda de `min_child_samples` (2026-10-05, `busqueda_min_child.py`)

Comparación pareada contra 1000 (misma CV GroupKFold por pitcher, entrenamiento 2024+2025):

| min_child_samples | log-loss media | dif pareada vs 1000 (media) | dif std | folds que gana 1000 |
|---:|---:|---:|---:|:---:|
| 20 | 1.43323 | +0.00032 | 0.00019 | 5/5 |
| 50 | 1.43315 | +0.00023 | 0.00017 | 4/5 |
| 100 | 1.43319 | +0.00027 | 0.00031 | 5/5 |
| 200 | 1.43308 | +0.00017 | 0.00012 | 5/5 |
| 500 | 1.43293 | +0.00002 | 0.00010 | 2/5 |
| **1000** | **1.43292** | 0 | 0 | — |
| 2000 | 1.43314 | +0.00023 | 0.00011 | 5/5 |

**Elegido: `min_child_samples=1000`** (en `PARAMETROS`). Es el de menor log-loss. 500 es indistinguible en la comparación pareada (1000 gana solo 2/5), pero entre iguales se prefiere el más regularizado → 1000. 2000 pierde en 5/5 folds (consistente), así que no se sube. La std pareada (~0.0001) es ~18× menor que la std entre folds (~0.0018): por eso el pareado distingue que 2000 es peor, algo invisible con la regla vieja (que daba todo "empatado").

_Nota: la búsqueda se corrió antes de añadir `SpinAxis` y la limpieza completa. El modelo es insensible a `min_child_samples` en todo el rango, así que la elección (1000) se mantiene; no se re-corrió._

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

#### Predicciones en los 12 conteos (`predecir_12_conteos.py`, insumo del Stuff+ — verificado 2026-10-05)

- Para cada pitch, con la física fija, se predice `P(k | física, conteo)` en los 12 conteos (`Balls × Strikes`). **Out-of-fold:** cada pitch de entrenamiento se predice con el modelo del fold que no vio a su pitcher, leyendo `../salidas/folds.csv` (escrito por `modelo_multiclase.py`) en vez de reconstruir la partición —así un pitcher no se asigna mal en silencio si cambia el orden o los datos. Salida: `../salidas/predicciones_12conteos.parquet` (formato largo, `p_*` float32; **5,937,648 filas = 494,804 × 12**). Falta combinar con `RV_k(conteo)` y los pesos de la liga en `scoring.py` para el xRV neutral de Stuff+.
- **Sanity verificados:** (1) las 5 probabilidades suman 1 (máx |err| ≈ 1e-7, redondeo float32); (2) en el conteo real coincide con `predicciones_oof.parquet` en los 494,804 pitches (dif. máx abs ≈ 3e-8) → features y asignación de folds bien armadas; (3) patrones con sentido: al subir strikes cae `called_strike` y suben los swings; 3-0 tiene `called_strike` ~52 % y whiff ~4 %; el slider mediano tiene más whiff que la four-seam en todo conteo (0-2: 15.7 % vs 9.5 %).

#### Stuff+ neutral (`scoring.py`, verificado 2026-10-05)

- `calcular_xrv(df, "neutral")`: xRV(pitch, conteo) = Σ_k P(k)·RV_k(conteo); xRV_Stuff = Σ_conteo w_conteo·xRV (promedio de los 12 conteos con pesos de la liga). `a_escala_plus(s, year)`: Stuff+ = 100 + 10·z **por temporada**, con s = −xRV_Stuff.
- v1 sobre el entrenamiento **out-of-fold** (2024+2025); `in_play` con RV_k **v0**; **sin altitud**.
- **Verificado:** (1) Stuff+ media 100 / sd 10 por temporada (241,196 pitches en 2024, 253,608 en 2025); (2) correlación a nivel pitcher Stuff+ vs tasa de whiff **r = 0.43** (n≥200, 599 pitchers) — positiva y fuerte, pero no 1, porque Stuff+ es más que whiffs; (3) **leaderboard con sentido:** arriba four-seams de 92–97 mph y rompientes de spin alto (2500–2880 rpm) con whiff 13–18 % (Stuff+ 117–124); abajo sinkers de poco spin (2000–2350) con whiff 2–9 % (Stuff+ 81–84).
- Salidas (contrato §6): `../salidas/scores_pitch.parquet` (`PitchUID, pitcher_anon_id, AutoPitchType, year, xrv_stuff, stuff_plus`), `scores_pitcher_tipo.csv`, `scores_pitcher.csv`.
- Pendiente: contexto `"real"` (Location+/Pitching+, Matías); RV_k de contacto (paso 7); altitud (paso 10); scores finales tras la validación (reentrenar con todo, incluir 2026).

### Paso 9 · Validación final

- Sobre el test sellado: log-loss/RMSE vs. baseline "xRV promedio por conteo"; curvas de calibración.
- Pitcher holdout (GroupKFold), season holdout (última temporada), park holdout (entrenar sin la categoría de altitud alta y probar en ella).
- Después de validar: reentrenar con todo para los scores finales.

#### Chequeo preliminar en el test sellado (2026-10-05)

No sustituye la validación final (esa será con el xRV/Stuff+ completo); es solo un sanity del modelo de resultado. El modelo (entrenado en 2024+2025) solo se lee, no se tunea nada contra el test.

- Modelo vs baseline por conteo en 2026 (130,656 lanzamientos): log-loss 1.4449 vs 1.4607 → **mejora 1.08 %**. En CV era 1.00 % → se sostiene fuera de muestra (no sobreajusta).
- **Generaliza a pitchers nuevos:** vistos en train 1.10 %, nuevos (no vistos) 1.05 % — casi iguales (robustez pitcher-holdout).
- whiff (log-loss binaria): 2.46 %. Calibración dentro de ~2 pp, con una leve deriva de temporada (2026 con algo más de strikes cantados y menos bolas que 2024-25).

### Paso 10 · Altitud

- La altitud actúa por dos canales: (1) menos movimiento (ya viene en IVB/HB medidos); (2) contacto más caro (lo aprende el modelo vía `altitude_category`).
- Delta de altitud = mismo pitch evaluado en "alta" vs "baja", cambiando la categoría y ajustando el movimiento: IVB_bajo ≈ IVB_alto · ρ_bajo/ρ_alto (idem HB). Factor a validar empíricamente con los mismos pitchers en distintas categorías (efecto fijo por pitcher).
- Densidad vs. altitud (atmósfera estándar, la que usa `altitud.py`): ρ/ρ₀ = (1 − 2.25577e−5·h)^4.25588. A 2,232 m, ρ ≈ 0.80 de la del nivel del mar (fórmula); la densidad medida por frenado en Harp Helú es 0.82. **Las fórmulas de física (Magnus, arrastre, reescalado de movimiento) están registradas y explicadas en [`docs/fisica_altitud.md`](fisica_altitud.md)** — para que Fran las investigue y valide.
- **Altitud de Harp Helú:** usamos **2,232 m** (`HARP_HELU_M` en `altitud.py`, fuente Mediotiempo/Milenio 4-jun-2025). El brief del ISAC dice "~2,240 m"; la diferencia (8 m) es despreciable y adoptamos 2,232 m por tener fuente citada en el código.
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

## 6. Estado (2026-10-05)

- **Hecho:** `limpieza.py` completa (bunts, `SpinAxis` y ángulos volteados, tabla de filtros), `objetivo.py`, `particion.py`, `modelo_multiclase.py` v1 (opción B + `SpinAxis`, `min_child_samples=1000`, sin altitud), `busqueda_min_child.py`, `predecir_12_conteos.py`, `run_values.py` (RV_k con corredores), `scoring.py` v1 (Stuff+ neutral), `estadios.py`/`altitud.py` (altitud continua). README, CLAUDE.md, guía y decisions al día.
- **Pendiente inmediato:** `contacto.py` (Paso 7, Matías); Location+/Pitching+ (Matías); contrafactual de altitud (Paso 10); validación final (Paso 9); scores finales (reentrenar con todo); pregunta a ISAC sobre altitud/estadios/clima.

## 7. Bitácora de cambios

- **2026-09-24:** arquitectura de 3 modelos; roles; estructura modular.
- **2026-09-24:** diccionario de ISAC revela que no hay estadio ni corredores → altitud por categoría; linear weights con outs.
- **2026-09-28:** opción B para el conteo; Pitching+ desde cero; Location+/Pitching+ en conteo real restando el promedio del conteo; altitud como feature + contexto de evaluación + ajuste de movimiento en el delta; test sellado = última temporada; `min_child_samples` por CV.
- **2026-09-28:** método de RV_k sin depender del orden de los lanzamientos (RE por outs con suma de `RunsScored` en `Outs` ≥ o); usar datos sin limpiar para valores de carrera.
- **2026-10-05:** Matías reconstruye estadios y densidad por juego → altitud continua y Harp Helú identificable.
- **2026-10-05:** `objetivo.py` corrido con datos reales (627,733 lanzamientos). Verificado: proporciones de las 5 clases (ball 37.8 %, called_strike 15.4 %, whiff 11.3 %, foul 17.2 %, in_play 18.3 %); `whiff` coincide 100 % con `is_whiff`; `is_swing` difiere en 1,119 filas (0.18 %), todas pitches tomados (BallCalled 457, StrikeCalled 390, HitByPitch 227, BallinDirt 45) que el ISAC marca como swing — se conservan porque el target se arma desde `PitchCall`.
- **2026-10-05:** `particion.py` escrito y corrido (Paso 4). Test sellado = temporada 2026: entrenamiento 496,625 lanzamientos / 829 pitchers (2024+2025), test 131,108 / 549 pitchers. De los pitchers del test, 301 (54.8 %) se vieron en entrenamiento y 248 (45.2 %) son nuevos.
- **2026-10-05:** `modelo_multiclase.py` v1 (opción B, sin altitud) corrido sobre entrenamiento (2024+2025). Mejora ~1.0 % sobre el baseline por conteo; la señal de la física se concentra en whiff (log-loss binaria −2.33 %), poco en contacto y nada en ball. Calibración buena salvo sesgo leve en 3-0 (±1 pp). Contrato OOF con `Balls`/`Strikes`; guardados 5 modelos de folds + final.
- **2026-10-05:** `predecir_12_conteos.py` (insumo del Stuff+, Paso 8): predice cada pitch en los 12 conteos out-of-fold, leyendo `../salidas/folds.csv` (nuevo, escrito por `modelo_multiclase.py`). Salida `../salidas/predicciones_12conteos.parquet` (largo, float32, 5.96M filas). Sanity: coincide con `predicciones_oof.parquet` en el conteo real (dif. máx ≈ 3e-8). Contrato §6 de la guía actualizado (nuevo artefacto + `folds.csv`).
- **2026-10-05:** añadidos `CLAUDE.md` y `docs/guia_proyecto.md` al repo y alineados con decisions.md: contrato OOF de `modelo_multiclase.py` (§6) ahora incluye `Balls`/`Strikes`; Estado de §6 actualizado (`particion.py` hecho; módulo 6 v1 hecho con `min_child_samples=1000`, v2 agrega `densidad_juego`); unificada la altitud de Harp Helú a **2,232 m** (fuente `altitud.py`; el brief dice ~2,240 m); marcada resuelta la pregunta abierta 9.2 de la guía (apunta al Paso 2).
- **2026-10-05:** **corregida la regla de desempate de `min_child_samples`** (Paso 6). La regla vieja ("si hay empate dentro de la desviación entre partes, elegir el más grande") estaba mal: la std de log-loss entre folds está dominada por la distinta dificultad de cada fold —común a todos los valores— así que no sirve de tolerancia y daba todo "empatado" (incluido 2000). Se cambió a comparación **pareada fold a fold** (ver regla corregida en Paso 6). Búsqueda (`busqueda_min_child.py`): elegido **`min_child_samples=1000`** (menor log-loss; 500 indistinguible pero menos regularizado; 2000 pierde 5/5 folds). `PARAMETROS` actualizado a 1000 y modelo re-corrido; OOF/modelos regenerados.
- **2026-10-05:** `limpieza.py` completo: bunts, volteo de `SpinAxis` (360 − ángulo), `HorzRelAngle` y `HorzApprAngle` para zurdos, tabla de filas quitadas por filtro (9,542 de 635,002 = 1.5 %) y `../pitches_clean.parquet`. En `modelo_multiclase.py` se agrega `SpinAxis` (mejora 1.04 % → 1.04 %, casi nada por sí solo) y se **descarta `VertRelAngle`** para Stuff+: codifica la ubicación (mejora 1.04 % → 12.35 %).
- **2026-10-05:** `run_values.py` (paso 5, Matías). El método solo con outs falla: una base por bolas valía +0.03 carreras (MLB ≈ +0.3) porque el corredor que llega a base no cambia los outs; por eso las bolas en 3-0 salían negativas. **Cambio:** las filas del parquet están en orden cronológico (outs no decrecen en 99.8 % de las medias entradas), así que se cuenta el número de corredores en base (bateadores que terminaron − outs − carreras) y se usa RE(outs, corredores), 12 estados. Resultados con 2024–2025: BB +0.42, 1B +0.54, 2B +0.69, 3B +0.84, HR +1.44, K −0.34, out en juego −0.36; pasan las 7 revisiones (RE baja con outs y sube con corredores, V sube con bolas y baja con strikes, bolas > 0, strikes < 0, pesos suman 1). Limitación: se sabe cuántos corredores hay, no en qué base.
- **2026-10-05:** Revisión del merge de `main` en `ricardo/objetivo-decisions` (Matías): en `modelo_multiclase.py` quedaron dos asignaciones de `FEATURES` y la segunda borraba `Balls`/`Strikes` (y `predecir_12_conteos.py` fallaba por `SpinAxis`). Corregido: `SpinAxis` va en `FISICA` y `FEATURES = FISICA + CONTEO`. Con eso: mejora 1.00 % sobre el baseline por conteo, calibración por conteo dentro de ~1 pp (3-0: 2.2 pp, n = 6,633). `altitud.py` corregido al exponente de densidad.
- **2026-10-05:** registradas y explicadas las fórmulas de física de la altitud en `docs/fisica_altitud.md` (densidad vs. altitud, Magnus → movimiento ∝ ρ, arrastre, `densidad_juego` medida, límites como el viento, y preguntas para que **Fran** las investigue). Alineada la fórmula de densidad: se usa la ley de potencias de la atmósfera estándar `(1 − 2.25577e−5·h)^4.25588` (como `altitud.py`), no la exponencial `e^(−h/8400)` que mencionaban versiones viejas.
- **2026-10-05:** `scoring.py` (Paso 8, Ricardo): xRV neutral (Σ_k P(k)·RV_k promediado sobre los 12 conteos con pesos de la liga) + escala Stuff+ = 100 + 10z por temporada. v1 sobre entrenamiento OOF, `in_play` con RV_k v0, sin altitud. Verificado: Stuff+ media 100 / sd 10 por temporada; correlación Stuff+ vs tasa de whiff r = 0.43 (nivel pitcher); leaderboard con sentido (four-seams rápidas de alto spin y rompientes de whiff alto arriba, Stuff+ 117–124; sinkers de poco spin abajo, 81–84). Salidas `scores_pitch.parquet`, `scores_pitcher_tipo.csv`, `scores_pitcher.csv`. Falta: contexto real (Location+/Pitching+), RV_k de contacto, altitud y scores finales.
- **2026-10-05:** re-corrido el pipeline completo sobre la limpieza nueva + `SpinAxis` y refrescados los números en decisions.md (Ricardo): total 625,460 (solo bajó `in_play` por los bunts → 18.0 %); partición train 494,804 / test 130,656 (829 / 549 pitchers, 301 vistos / 248 nuevos); modelo mejora CV 1.00 %, whiff 2.30 %; `predecir_12_conteos` 5,937,648 filas con sanity OK (coincide con OOF en el conteo real, dif. 3e-8); chequeo preliminar en test 2026: mejora 1.08 %, generaliza a pitchers nuevos (1.10 % vistos / 1.05 % nuevos). Alineado el cuerpo del Paso 5 al método de corredores de `run_values.py` (antes describía el de solo outs).
