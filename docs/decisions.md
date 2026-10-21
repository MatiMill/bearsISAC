# Decisiones y especificación del proyecto · bearsISAC

**Hackathon ISAC 2026 · Stuff+ para la LMB / Diablos Rojos (Harp Helú, 2,232 m)**
Entrega: **2026-10-15** (meta interna 14 oct) · Finalistas 20 oct · Presentaciones 10–11 nov

Este documento es la **fuente de verdad**: si una decisión cambia, se cambia aquí con fecha y motivo.
_Última actualización: 2026-10-21 (auditoría de correctitud; ver la bitácora)._

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

- Quitar no-competencia: `PitchCall` ∈ {BallIntentional, AutomaticBall, AutomaticStrike, Undefined}; bunts (`hit_type` = Bunt; NaN se queda) — hecho.
- Quitar no describible: `AutoPitchType` = Other; mano desconocida; nulos en física clave.
- Quitar errores: `RelSpeed` ∉ [60, 105]; `Extension` ∉ [3, 8]; |IVB| o |HB| > 30.
- Voltear zurdos: `HorzBreak`, `RelSide`, `HorzRelAngle`, `HorzApprAngle` × (−1); `SpinAxis` → 360 − `SpinAxis` — hecho (después de voltear, las medianas de zurdos y derechos coinciden). Después: negativo = hacia el guante siempre.
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

> **Correcciones 2026-10-21 (auditoría):** (1) el precio de `in_play` ya **no incluye los bunts** (la clase `in_play` del modelo los excluye en `limpieza.py`; los bunts siguen en V y RE, son turnos reales): el precio en 0-0 sube +0.010. (2) La 4.ª bola se paga con el valor de las bases por bolas que terminaron en **ese** conteo (3-s) y el 3.er strike con el de los ponches que terminaron en b-2 (antes, el promedio de toda la liga; cae a ese promedio si hay < 50 turnos). (3) La clase `ball` incluye los pelotazos: su precio mezcla el de una bola y el de un pelotazo (+0.44 carreras) según la fracción de pelotazos entre los pitches `ball` de ese conteo (antes, cada pelotazo valía como una bola, ≈ 0.4 carreras de menos). (4) En `valor_de_turnos`, `PitchCall = HitByPitch` se revisa antes que `KorBB = Walk` (22 pitches con ambos). Efecto: ningún precio cambia más de 0.011; revisión de consistencia Σ_k P(k | conteo)·RV_k con las frecuencias reales: máx |desviación| **0.0094 → 0.0059** (3-0 de −0.0094 a −0.0054).

- Usar los datos **sin limpiar** (los filtros quitan filas con carreras y resultados). Solo temporadas de entrenamiento (2026 sellado).
- Media entrada = (`game_anon_id`, `Inning`, Top/Bottom). Turno = (media entrada, `batter_anon_id`); un turno nuevo empieza después de cada lanzamiento final o al cambiar el bateador (no se descartan turnos).
- Lanzamiento final del turno: `KorBB` ∈ {Strikeout, Walk} o `PitchCall` ∈ {InPlay, HitByPitch}.
- **RE(outs, corredores):** carreras promedio desde ese estado hasta que termina la media entrada. RE = 0 al terminar la entrada.
- Valor de cada final de turno = RE(estado al terminar) − RE(estado al empezar) + `RunsScored` del turno.
- V(conteo) = promedio del valor final de los turnos que pasaron por ese conteo (pares únicos turno-conteo, porque los fouls con 2 strikes repiten conteo).
- RV_k(b, s), perspectiva del bateador (+ = bueno para el bateador):
  - ball: V(b+1, s) − V(b, s); si b = 3: V_walk − V(3, s).
  - called_strike, whiff: V(b, s+1) − V(b, s); si s = 2: V_K − V(b, 2).
  - foul: igual que strike si s < 2; 0 si s = 2.
  - in_play (v0): promedio del valor de los InPlay (sin bunts) en ese conteo − V(b, s). Se reemplaza en el paso 7 por E[valor | contacto, pitch].
- **Resultados verificados (2024–2025):** linear weights BB +0.42, 1B +0.54, 2B +0.69, 3B +0.84, HR +1.44, K −0.34, out en juego −0.36 (cerca de MLB). Pasan las 7 revisiones: RE baja con outs y sube con corredores, V sube con bolas y baja con strikes, toda bola > 0, todo strike < 0, los pesos suman 1.
- Salidas (contrato §6): `../salidas/rv_tabla.csv` (`balls, strikes, outcome_class, rv`), `valores_evento.csv`, `pesos_conteo.csv`.
- Aproximaciones: se sabe **cuántos** corredores hay, no en qué base; HBP dentro de la clase ball (con precio de pelotazo mezclado, ver arriba); las carreras en pitches no finales (wild pitch, robo) se suman al valor del turno (2.9 % de las carreras).
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
- **Re-corrido 2026-10-21** con los RV_k corregidos: Stuff+ vs whiff r = 0.43 (igual). `scoring.py` ahora comprueba `validate="one_to_one"` y que no falte el año (antes habría dado NaN en silencio).
- **Escala por pitcher (2026-10-21):** los scores por pitch tienen sd 10 **entre pitches**, pero el promedio de un pitcher varía mucho menos: Stuff+ sd ≈ 4.5 entre pitchers con muestra, Location+/Pitching+ sd ≈ 0.9 (solo ≈ 1 % de la varianza entre pitches es del pitcher; el resto es dónde cayó cada pitch). No es un error del modelo (comprobado: `min_child_samples`, features y folds no comprimen nada; los scores se recalculan exactos). Para comparar pitchers, `scoring.a_escala_pitcher` reescala el promedio a **100 + 10z entre pitchers con ≥ 300 pitches** (columnas `stuff_plus_pitcher`, `location_plus_pitcher`, `pitching_plus_pitcher` en `scores_*pitcher.csv`; el orden no cambia). Los scores por pitch no cambian.
- Salidas (contrato §6): `../salidas/scores_pitch.parquet` (`PitchUID, pitcher_anon_id, AutoPitchType, year, xrv_stuff, stuff_plus`), `scores_pitcher_tipo.csv`, `scores_pitcher.csv` (+ `stuff_plus_pitcher`).
- Pendiente: RV_k de contacto (paso 7); altitud (paso 10); scores finales tras la validación (reentrenar con todo, incluir 2026). El contexto `"real"` ya está (ver Location+ abajo).

#### Location+ (`zona.py`, `modelo_location.py`, `scoring_location.py`, Matías — verificado 2026-10-06)

Pregunta: **¿la puso donde debía?** Mismo método que Stuff+ (LightGBM multiclase → xRV → 100 + 10z); cambian las features y el contexto de evaluación (conteo real). v1 sin altitud (la tabla de la sección 3 la deja opcional).

**Ubicación (`zona.py`).**
- `lado_relativo` = `PlateLocSide` con el signo del bateador (× −1 para zurdos): **+ = adentro**, − = afuera, para cualquier bateador. `PlateLocSide` no se voltea por la mano del pitcher en `limpieza.py` (es la vista del catcher). Verificado con los pelotazos: mediana +2.0 ft (zurdos) y +2.03 ft (derechos).
- `zona`: rectángulo |`PlateLocSide`| ≤ 0.7083 ft (17" / 2) y 1.62 ≤ `PlateLocHeight` ≤ 3.21 ft, en 3 × 3 (1 alto-adentro … 9 bajo-afuera) + 10 = fuera. Coincide con `in_strike_zone` del ISAC en el **99.999 %**. Solo para reportes y el dashboard: el modelo usa las coordenadas continuas. % cantado strike en pitches tomados: 95–99 % en las filas alta y media, 81–88 % en la baja, 14.1 % fuera.
- Se quitan los pitches con `BatterSide` desconocido (Undefined, Switch o nulo; tras la limpieza no queda ningún Switch): 301 en total (0.05 %), 133 del entrenamiento.

**Modelo (`modelo_location.py`).**
- Features: `PlateLocHeight`, `lado_relativo`, `Balls`, `Strikes`, `bateador_zurdo`, `misma_mano` (las dos juntas distinguen los 4 emparejamientos pitcher/bateador), `tipo_pitch` (categórica de `AutoPitchType`, categorías fijas `TIPOS_VALIDOS`). Sin física: eso es Stuff+. `VertRelAngle` no entra (la sección 3 define Location+ como ubicación + conteo + lado + tipo; queda para Pitching+).
- **Misma partición que Stuff+:** lee `../salidas/folds.csv` (`validacion_cruzada(..., fold_de_pitcher=...)`, parámetro nuevo y opcional en `modelo_multiclase.py`; sin él sigue usando GroupKFold igual que antes). Motivo: al quitar 133 filas, GroupKFold reparte distinto y **59 % de los pitches** habría caído en otro fold que en Stuff+; con la misma partición, las predicciones OOF de Stuff+, Location+ y Pitching+ son comparables pitch a pitch. Un `assert` lo comprueba.
- **`min_child_samples = 2000`** (`PARAMETROS_LOCATION`; el resto igual que Stuff+). Búsqueda (`busqueda_min_child_location.py`, misma regla pareada que el Paso 6):

| min_child_samples | log-loss media | dif pareada vs 1000 (media) | dif std | folds que gana 1000 |
|---:|---:|---:|---:|:---:|
| 20 | 1.02479 | +0.00099 | 0.00018 | 5/5 |
| 50 | 1.02468 | +0.00087 | 0.00016 | 5/5 |
| 100 | 1.02467 | +0.00086 | 0.00016 | 5/5 |
| 200 | 1.02420 | +0.00039 | 0.00021 | 5/5 |
| 500 | 1.02410 | +0.00030 | 0.00023 | 5/5 |
| 1000 | 1.02381 | 0 | 0 | — |
| **2000** | **1.02386** | +0.00006 | 0.00031 | 3/5 |

  1000 tiene la menor log-loss, pero 2000 es indistinguible (1000 gana solo 3/5) → por la regla, el más regularizado: **2000**. Como 2000 es el borde de la malla, se probó también 5000: pierde contra 2000 en **5/5 folds** (+0.0011), así que no se sube más.

**Resultados verificados** (entrenamiento 2024+2025: 494,671 pitches, 829 pitchers, folds de Stuff+):
- **Mejora sobre el baseline por conteo: 29.18 %** de log-loss (28.91–29.42 % entre folds; modelo ≈ 1.024 vs baseline ≈ 1.446). Stuff+: 1.00 %.
- Por clase (log-loss binaria): **ball 52.7 %**, called_strike 24.3 %, in_play 17.7 %, foul 12.5 %, whiff 9.4 %. Complementa a Stuff+: la ubicación explica ball/strike; la física, el whiff.
- Calibración: general ≤ 0.02 pp por clase; por conteo máx 0.98 pp (ninguna celda ≥ 1 pp, también en 3-0); por zona máx 0.29 pp (fuera de la zona: ball 57.58 % real vs 57.55 % predicho).
- Modelos recargados desde `.txt` reproducen el OOF exacto (dif. 0.0; la categórica `tipo_pitch` se guarda bien).

**Score (`scoring.py` contexto `"real"` + `scoring_location.py`).**
- `calcular_xrv(df, "real")`: xRV = Σ_k P(k)·RV_k(conteo real); `xrv_vs_conteo` = xRV − xRV promedio de la liga en ese conteo (tabla guardada en `xrv_promedio_conteo_location.csv`, para calificar pitches nuevos con `promedio_conteo=`). Location+ = `a_escala_plus(−xrv_vs_conteo, year)`. El contexto `"neutral"` no cambió: reproduce `scores_pitch.parquet` exacto (dif. 0.0).
- El xRV promedio por conteo sale casi 0 (−0.0005 a +0.0026) salvo 3-0 (−0.0079) y 3-1 (−0.0060). Lo esperado es ≈ 0: V(conteo) ya es el valor esperado desde ese conteo, así que el siguiente pitch en promedio no lo mueve. La desviación **no es del modelo**: el valor real promedio de lo que pasó (RV_k del resultado observado) también es negativo (3-0 −0.0054, 3-1 −0.0059). Viene de las aproximaciones de `run_values.py` (corredores sin base exacta, `in_play` v0). Antes de las correcciones de `run_values.py` del 2026-10-21 era −0.012 (3-0) y −0.007 (3-1); la parte que queda es la de estas aproximaciones. Restar el promedio del conteo lo neutraliza en Location+.
- **Sanity verificados:** (1) media 100 / sd 10 por temporada (241,166 pitches en 2024, 253,505 en 2025); (2) cada conteo centrado en 0 tras restar (máx |media| ≈ 4e-18); (3) **la buena ubicación depende del conteo**: en 3-0 dentro de la zona 113–116 y fuera 88.3; en 0-2 el centro de la zona es lo peor (97.2) y las esquinas bajas/afuera lo mejor (101.6–102.7); (4) a nivel pitcher (n ≥ 200, 599 pitchers) Location+ vs tasa de bolas **r = −0.74**; (5) Location+ vs Stuff+ v3 **r = −0.03** (v1: 0.04) (miden cosas distintas, como se buscaba). Estabilidad 2024 → 2025 (181 pitchers con ≥ 200 pitches en cada año): Location+ **r = 0.62**, Stuff+ r = 0.78. Stuff+ es más estable que Location+, como se ve también en MLB: el comando varía más de año a año que el "stuff".
- Salidas (contrato §6 de la guía): `../salidas/predicciones_oof_location.parquet`, `validacion_location.csv`, `scores_location_pitch.parquet`, `scores_location_pitcher_tipo.csv`, `scores_location_pitcher.csv`, `xrv_promedio_conteo_location.csv`; `../modelos/modelo_location_fold{1..5}.txt` + `modelo_location.txt`.
- Pendiente: `in_play` con RV_k de contacto (paso 7); validación en el test sellado (paso 9); scores finales tras reentrenar con todo; altitud opcional.

#### Pitching+ (`pitching_plus.py`, Ricardo — verificado 2026-10-12)

Ve **todo**: física + conteo + ubicación + lado del bateador (sin altitud por ahora). Mismo motor que Stuff+/Location+ (`validacion_cruzada` con `features=` y los folds de `folds.csv`).

**Modelo (`pitching_plus.py`).**
- Features: 8 físicas (incl. `SpinAxis`) + `Balls`, `Strikes` + `PlateLocHeight`, `lado_relativo` + `bateador_zurdo`, `misma_mano`. **Sin `tipo_pitch`** (la física ya codifica la forma del pitch, coherente con Stuff+) ni altitud. `FEATURES_PITCHING` es una lista para añadir `densidad_juego` en v2. `min_child_samples = 1000` (igual que Stuff+).
- Misma partición que Stuff+/Location+ (`folds.csv`), comprobada con un `assert`. Corre sobre 494,671 pitches (descarta los sin lado de bateador conocido, como Location+).
- **Verificado (entrenamiento 2024+2025):** mejora sobre el baseline por conteo **30.3 %** (Stuff+ 1.0 %, Location+ 29.2 %) — casi todo es la **ubicación**. Por clase (log-loss binaria): **ball 53.6 %** (Stuff+ 0.02 %), called_strike 25.3 %, in_play 18.6 %, foul 12.9 %, whiff 11.3 %. Calibración general ≤ 0.04 pp; por conteo máx 1.13 pp (3-0).
- Interpretación: Pitching+ predice el resultado mucho mejor porque **sabe dónde terminó el pitch**; no es "mejor" que Stuff+, responde otra pregunta (ejecución total vs. calidad intrínseca de la bola). El salto 1 % → 30 % es la ubicación, justo lo que Stuff+ excluye a propósito.
- Salidas: `../salidas/predicciones_oof_pitching.parquet` (mismas columnas que el OOF de Stuff+), `validacion_pitching.csv`; `../modelos/modelo_pitching_fold{1..5}.txt` + `modelo_pitching.txt`.

**Score (`scoring_pitching.py`, contexto `"real"`, verificado 2026-10-12).**
- `calcular_xrv(oof, "real")` (xRV en el conteo real − promedio del conteo) + `a_escala_plus(−xrv_vs_conteo, year)`. Salidas: `../salidas/scores_pitching_pitch.parquet` (`…, xrv_pitching, xrv_vs_conteo, pitching_plus`), `scores_pitching_pitcher_tipo.csv`, `scores_pitching_pitcher.csv`, `xrv_promedio_conteo_pitching.csv`.
- **Verificado:** media 100 / sd 10 por temporada; cada conteo centrado en 0 (máx |media| 4e-18); a nivel pitcher (n ≥ 200, 599) Pitching+ vs tasa de bolas **r = −0.73** y vs whiff **r = +0.27** → combina los dos canales.
- **Chequeo de la guía §4:** Pitching+ correlaciona con **Stuff+ v3 (r = 0.28; con v1 era 0.42)** y con **Location+ (r = 0.87)**, mientras Stuff+ y Location+ son casi independientes (r = −0.03). (Los r de las secciones de Location+ y Pitching+ medidos antes del 2026-10-21 usaban Stuff+ v1; las revisiones cruzadas de `scoring_location.py` y `scoring_pitching.py` leen ahora el Stuff+ v3 oficial si ya se corrió `contrafactual_altitud.py`.) Es decir, Pitching+ es la ejecución combinada, **dominada por la ubicación** (que mueve el ball/strike, la mayor parte del valor) con el stuff como aporte secundario — consistente con que la mejora saltó a 30 % al meter ubicación.
- Estabilidad 2024 → 2025 **r = 0.60** (Stuff+ v3 0.78, Location+ 0.62): el comando/ejecución se repite menos año a año que el stuff puro.

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
- (**Superado el 2026-10-14 y refinado el 2026-10-21:** el ajuste usa el efecto medido por tipo con `dx/dρ = g0 + κ·x` (exponencial, invertible), no la proporción ρ; ver "Implementación de la altitud" abajo.) Delta de altitud = mismo pitch evaluado en "alta" vs "baja", cambiando la categoría y ajustando el movimiento: IVB_bajo ≈ IVB_alto · ρ_bajo/ρ_alto (idem HB). Factor a validar empíricamente con los mismos pitchers en distintas categorías (efecto fijo por pitcher).
- Densidad vs. altitud (atmósfera estándar, la que usa `altitud.py`): ρ/ρ₀ = (1 − 2.25577e−5·h)^4.25588. A 2,232 m, ρ ≈ 0.80 de la del nivel del mar (fórmula); la densidad medida por frenado en Harp Helú es 0.82. **Las fórmulas de física (Magnus, arrastre, reescalado de movimiento) están registradas y explicadas en [`docs/fisica_altitud.md`](fisica_altitud.md)** — para que Fran las investigue y valide.
- **Altitud de Harp Helú:** usamos **2,232 m** (`HARP_HELU_M` en `altitud.py`, fuente Mediotiempo/Milenio 4-jun-2025). El brief del ISAC dice "~2,240 m"; la diferencia (8 m) es despreciable y adoptamos 2,232 m por tener fuente citada en el código.
- Hipótesis del brief: H1 (spin pierde en altura), H2 (cambio gana relativo), H3 (groundball pitchers valen más), H4 (spin rate solo es insuficiente; usar eficiencia).

#### Implementación de la altitud (`efecto_altitud.py` … `contrafactual_altitud.py`, Matías — verificado 2026-10-14, **corregido y re-corrido 2026-10-21**)

> **Correcciones de la auditoría del 2026-10-21** (detalle en la bitácora): (1) **el movimiento perdido se contaba dos veces** en el escenario Harp Helú del contrafactual y del simulador; (2) el ajuste de movimiento pasa de un γ único por tipo a `dx/dρ = g0 + κ·x` (forma exponencial, invertible); (3) Sweeper usa el efecto del Slider (antes caía en el promedio de la liga, de signo contrario en HB); (4) la densidad de referencia y la identificación de Harp Helú usan **solo las temporadas de entrenamiento** (`RHO_HARP = 0.830`); (5) el precio de la bola en juego excluye los bunts. Los números de abajo son los del re-corrido; las cifras del 2026-10-14 quedan superadas.

La altitud entra por **tres canales**, cada uno medido o validado por separado (no solo como una columna más de LightGBM, que no basta:
hay ~20 estadios, Harp Helú es la casa de los Diablos —equipo y estadio se mezclan— y el movimiento medido ya trae la altitud adentro).
Solo temporadas de entrenamiento; el test sellado (2026) no se usó.

**1. Efecto medido en la LMB (`efecto_altitud.py`, Idea 1).** Dentro de cada pitcher × tipo (efecto fijo; separa talento y equipo del estadio),
`x = α + γ·ρ`, con ρ = `densidad_juego`; IC 95 % por bootstrap de pitchers (300). 482,365 pitches, 766 pitchers. Controles: SpinRate no cambia
con ρ (elasticidad β = 0.00 [−0.01, 0.01]) y RelSpeed casi nada (+0.20 mph en Harp Helú) → el efecto de IVB/HB no está contaminado.

| Tipo | IVB, Harp Helú − mar (in) | HB, Harp Helú − mar (in) |
|---|---|---|
| Four-Seam | **−3.65** [−3.73, −3.57] | −2.36 |
| Sinker | −2.12 | −3.79 |
| Cutter | −2.29 | −0.43 |
| Changeup | −1.87 | **−3.94** |
| Slider | −0.74 | +1.22 (menos movimiento hacia el guante) |
| Curveball | +1.50 (menos caída) | +2.82 |
| Splitter | usa el efecto del Changeup (61 pitchers; IC de IVB que cruza 0) | idem |
| Sweeper | usa el efecto del Slider (< 30 pitchers, sin fila propia) | idem |

(Los Δ son del γ aditivo promedio por tipo, a ρ = 0.830; el ajuste por pitch usa `g0` y `κ`, abajo.)

- **El movimiento cae más que proporcionalmente a la densidad:** elasticidad IVB β ≈ 1.35 [1.31, 1.40] (HB 1.59), no 1.0 como dice la física simple (§3 de
  `fisica_altitud.md`). Se usa el efecto medido y no la proporción IVB·ρ_destino/ρ_origen. **Cambia el método previsto en el Paso 10** (que decía IVB_bajo ≈ IVB_alto · ρ_bajo/ρ_alto).
- **H1 (el spin pierde en altura): no.** El spin rate no cambia; lo que cae es el movimiento. **H2:** el cambio pierde menos que el promedio (ver delta relativo abajo).
- **Un γ único no basta (auditoría):** el cambio crece con cuánto se mueve el pitcher. Ajustando `Δx = (g0 + κ·x̄)·Δρ` dentro de pitcher × tipo, κ es distinto de 0 en casi todos los tipos y variables (p. ej. Sinker IVB κ = 2.23 [1.83, 2.55], Curveball IVB 1.98, Slider HB 2.21; Four-Seam IVB 0.84) y explica 1–3 % más de la varianza de movimiento dentro del pitcher. Con un solo γ, 22 % de los cutters (HB) y 11 % de los sliders (IVB) se movían en sentido contrario al físico.

**2. Contrafactual del movimiento (`altitud_movimiento.py`, Idea 2).** `ajustar_movimiento(ivb, hb, rho_origen, rho_destino, tipo=None)`: resuelve `dx/dρ = g0 + κ·x` (con `y = x + g0/κ`, `dy/dρ = κ·y`):

    x_destino = (x_origen + g0/κ) · exp(κ · (ρ_destino − ρ_origen)) − g0/κ

Es **exactamente invertible y componible** (mar → Harp → mar devuelve el pitch; origen → destino = origen → mar → destino; error ≈ 1e-14), y con κ ≈ 0 se reduce a la forma aditiva. Es la forma que necesitan `normalizar_movimiento` (`IVB_n`, `HB_n`: movimiento a nivel del mar, ρ = 1.00) y el simulador.
Verificado: tras normalizar, la pendiente dentro de pitcher × tipo contra ρ es ≈ 0 en los 6 tipos principales (|pendiente| ≤ 0.3 in por 1.00 de ρ, contra 4–23 sin normalizar), también por tercil de cuánto se mueve el pitcher. Límite: κ y g0 se estiman por tipo y variable, no por pitcher.

**3. Precio de la bola en juego (`precio_in_play_altitud.py`, Idea 3 simple).** Valor del turno que termina en juego (el de `run_values.py`, **sin bunts**) ~ efecto fijo por **bateador**
+ ρ + conteo + estado (outs, corredores), EE agrupados por bateador; 89,609 turnos, 659 bateadores. **δ = −0.366 carreras por 1.00 de densidad** [−0.420, −0.312]:
un batazo en juego cuesta **+0.062 carreras más en Harp Helú que a nivel del mar** [+0.053, +0.071]. El precio de `in_play` pasa a RV_inplay + δ·(ρ − ρ_medio)·P(in_play), con ρ_medio = 0.925.
Límites: un solo δ para todos los batazos (no hay modelo de contacto: **H3 sin probar**); lineal en ρ; extrapola de ρ medio 0.925 a 1.00.

**Stuff+ v2 y v3 (`modelo_stuff_altitud.py`, `arsenal.py`, `modelo_stuff_arsenal.py`).** Mismos folds (`folds.csv`) y parámetros que v1, log-loss pareada. Re-entrenados el 2026-10-21 con el movimiento normalizado nuevo.
- **v2** = v1 con `IVB_n`, `HB_n` (movimiento a nivel del mar) + `densidad_juego`: mejora 1.01 % (v1 1.00 %). Calibración por aire en Harp Helú: sobre-predice bola en juego (18.90 % vs 18.35 % real) y sub-predice foul (16.84 % vs 17.48 %).
- **v3 (Stuff+ oficial)** = v2 + arsenal (Idea 4): diferencias contra la recta primaria del pitcher en la temporada (velocidad, IVB_n, HB_n, release) y % de uso del tipo; con movimiento normalizado, son neutrales a la altitud.
  Mejora **1.25 %** (whiff 2.76 %); **v3 gana a v2 en 5/5 folds** (dif. pareada +0.0034, std 0.0007). Pitchers sin recta: 0 %.
- **Sesgo de v1 corregido:** pitches lanzados en Harp Helú: Stuff+ v1 = 97.2 vs 100.2 en el resto; v3 (nivel del mar) = 99.3 vs 100.1.
- **El escenario Harp Helú del contrafactual es la entrada natural del modelo** (movimiento a nivel del mar + densidad del juego): en los 29,446 pitches reales de Harp Helú predice whiff 10.84 % (real 11.01 %) y bola en juego 19.00 % (real 18.35 %); con el movimiento además bajado (lo que hacía el escenario anterior) predice 10.46 % y 19.71 %.

**Park holdout (`park_holdout.py`, Idea 5).** Entrenar sin el estadio y probar en él, mismos parámetros, solo entrenamiento:

| Prueba | v1 (sin altitud) | v2_mov (mov. normalizado) | v2 (+ densidad) |
|---|---|---|---|
| A. sin Harp Helú (29,446 pitches de prueba) | 0.89 % | **0.99 %** | 0.95 % |
| B. sin altura extrema (138,490) | **1.01 %** | 0.94 % | 1.00 % |

El método generaliza (mejora ≈ 1 % también en estadios no vistos; sin colapso). El movimiento normalizado mejora a v1 al dejar fuera Harp Helú (A) y no la mejora al dejar fuera toda la altura extrema (B): la ventaja es chica y **no concluyente**; la **densidad como feature también es inconclusa** (ayuda en B, empeora en A: sesgo in_play +0.77 pp).
Por eso el canal "resultado" se reporta aparte y marcado como el menos validado. v3 no se corrió en park holdout (las features de arsenal son por pitcher × temporada). Nota: γ se estima incluyendo el estadio dejado fuera (optimismo leve).

**Contrafactual (`contrafactual_altitud.py`).** Cada pitch (out-of-fold) evaluado en dos escenarios, promediando los 12 conteos:
`mar` = (IVB_n, HB_n, ρ = 1.00); `harp` = **el mismo movimiento normalizado con ρ = 0.830**. El modelo se entrenó con movimiento a nivel del mar más `densidad_juego`, así que la densidad ya trae el costo del movimiento perdido: bajar además el movimiento (como hacía la versión del 2026-10-14) **lo contaba dos veces**. Stuff+ = 100 + 10z con μ, σ del escenario *mar* por temporada (σ = 0.0064 carreras por pitch); `harp` usa los mismos μ, σ. El delta se separa en canales (suman exacto):
`movimiento` = modelo con movimiento bajado a Harp Helú (y las diferencias contra la recta bajadas) y ρ = 1.00 menos el escenario mar; `resultado` = (harp − mar) − movimiento; `precio` = el desplazamiento del precio de la bola en juego.

| Tipo | Stuff+ mar | Δ movimiento | Δ resultado | Δ precio | **Δ total** | **Δ relativo** |
|---|---|---|---|---|---|---|
| Four-Seam | 100.8 | −6.9 | −0.2 | −17.2 | −24.3 | **−2.4** |
| Sinker | 94.8 | −2.9 | −2.5 | −21.0 | −26.4 | −4.5 |
| Changeup | 95.9 | −1.1 | −1.8 | −18.3 | −21.1 | +0.8 |
| Cutter | 96.4 | −0.6 | −1.3 | −18.8 | −20.7 | +1.2 |
| Slider | 106.2 | −1.4 | −0.1 | −16.7 | −18.1 | +3.8 |
| Splitter | 96.7 | −0.7 | 0.0 | −16.1 | −16.8 | +5.1 |
| Curveball | 99.4 | −1.7 | +0.9 | −14.6 | −15.3 | +6.6 |
| Sweeper | 97.4 | −2.8 | +2.0 | −13.8 | −14.5 | +7.4 |
| **Todos** | 100.0 | −3.5 | −0.7 | −17.7 | **−21.9** | 0 |

- **Lectura:** un pitch promedio pierde ≈ 22 puntos en Harp Helú (antes 26), casi todo (−17.7) por el precio de la bola en juego: un parque con mucha carrera, un efecto casi uniforme (**no es "stuff"**). Lo que distingue un tipo de otro es el **delta relativo** (cuánto más o menos pierde que el promedio): las **rectas pierden más** (Four-Seam −2.4, sobre todo por movimiento: −6.9; antes −5.0) y los rompientes y el splitter pierden menos (Curveball +6.6, antes +8.6). Entre pitchers (n ≥ 200) el Δ total varía con sd 1.4 [−26.2, −17.5].
- **Stuff+ de Harp Helú en su propia escala:** en la escala del mar un pitch promedio saldría ≈ 78 en Harp Helú y parecería malo cuando es el parque. `stuff_plus_harp_centrado` = `stuff_plus_harp` − Δ medio (100 = pitch promedio de Harp Helú); el dashboard muestra ese número y el Δ relativo.
- **H3 sin probar.** El sinker (pitch de rolling) pierde *más* que el promedio (probablemente por su mayor exposición a bola en juego, su Δ de precio es −21.0 vs −17.7; es una inferencia, no se midió); probar H3 requiere el modelo de contacto (Paso 7).
- `xrv_neutral` reproduce `scoring.py` (dif. 1e-9). El simulador del dashboard reproduce este contrafactual **exacto** (dif. máx 0.0 en 40 pitches, con el modelo de fold y μ, σ por año).

**Dashboard (Paso 11): rediseñado y probado el 2026-10-21** (`dashboard.py` y `simulador_altitud.py`). Cinco secciones con navegación lateral (Resumen, Candidatos, Laboratorio de pitches, Efecto de la altura, Metodología y glosario), texto de ayuda en cada número, nota "cómo leerlo" en cada pantalla, gráfica de rating con margen de incertidumbre y estilo sobrio (sin emojis). Probado de forma automática en todas las secciones y entradas (tipo de pitch, mano, aire, filtros, pitchers); no se pudo capturar una imagen del renderizado, así que el aspecto visual no se verificó con una captura. El simulador promedia los 5 modelos de fold (la calibración viene de predicciones out-of-fold; el modelo final solo agregaba ±2.6 puntos de ruido por pitch), pide la recta primaria para los pitches que no son rectas (en el entrenamiento la referencia siempre es Four-Seam, Sinker o Cutter) y acepta zurdos con mediciones del radar sin voltear (`mano="L"`).

**Limitaciones a decir en el documento:** (1) κ y g0 por tipo, no por pitcher; (2) canal "resultado" inconcluso; (3) un solo δ para los batazos y sin modelo de contacto (H3, H4 —eficiencia de spin— sin hacer); (4) δ comparado dentro de bateador, no controla pitcher ni clima;
(5) `densidad_juego` incluye la temperatura y Harp Helú se identifica por récord (hay otros dos estadios con ρ ≈ 0.83); (6) 19 pitches (2 juegos de 2024) sin densidad medida usan la media de su estadio; (7) el Δ en puntos depende de una σ pequeña (0.0064 carreras por pitch): el efecto de parque del precio de la bola en juego (≈ 0.011 carreras por pitch) ≈ 1.8σ;
(8) la **reconstrucción de equipos/estadios es ruidosa** (p. ej. `estadio_03` tiene 3 / 13 / 17 juegos por temporada: un equipo mal agrupado; `altitud.py` le asigna 2,192 m por ranking). Harp Helú (`estadio_05`: 125 juegos, ρ 0.82–0.83 cada año) sí es sólido; la densidad se mide por juego, así que el error solo afecta a las etiquetas de estadio (ajuste por parque del ranking); (9) la densidad medida de 2026 sube ≈ 2 % en estadios sin altitud (¿temporada parcial, clima o cambio del radar?): hay que revisarlo antes de aplicar v3 al test sellado; (10) en 2026 se reconstruyen los equipos con datos de 2026 (clustering), pero la referencia de densidad y la identificación de Harp Helú ya no usan sus resultados.

#### A quién contratar (`contratacion.py`, Matías — verificado 2026-10-14, **corregido y re-corrido 2026-10-21**)

Ranking de pitchers para Harp Helú, en **carreras por 100 pitches contra un pitcher promedio** (+ = mejor): `rating = ejecución + ajuste de altitud`.
- **Ejecución:** −xRV_vs_conteo medio de **Pitching+** (2024+2025). Se **encoge** hacia un promedio previo con la confiabilidad R = var_real / (var_real + var_ruido / n). Correcciones del 2026-10-21 (auditoría):
  - **Los pitches de un mismo juego no son independientes.** Antes `var_ruido` era la varianza de un pitch suelto (3.18e-3) y R(1,000 pitches) salía 0.88. Ahora `var_real` (1.96e-5) sale de la covarianza entre dos mitades al azar de los **juegos** de cada pitcher y el ruido efectivo (4.07e-3) de la diferencia entre mitades (donde el talento se cancela). R(1,000) = **0.83**. Comprobado contra la correlación observada entre mitades de juegos (bandas de ≈ 220 / 390 / 615 / 1,030 pitches por mitad): observada 0.54 / 0.65 / 0.74 / 0.76; el método nuevo da 0.52 / 0.65 / 0.75 / 0.83 y el anterior 0.62 / 0.74 / 0.82 / 0.88 (siempre de más). En la banda de más uso el método nuevo sigue algo optimista: la incertidumbre es un **mínimo**.
  - **El promedio previo depende de la muestra:** los pitchers con pocos pitches son peores en promedio (relevistas, llamados de AAA). Previo = a + b·log(n): −0.27 carreras/100 pitches con 300 pitches, +0.04 con 1,000, +0.27 con 2,500 (antes, 0 para todos: favorecía ≈ 0.04 a los de 300–500 pitches).
  Incertidumbre = √(var_real·(1−R)); no incluye que el talento cambie de un año a otro (la estabilidad real de Pitching+ 2024→2025 es r = 0.60) ni la del ajuste de altitud.
- **Ajuste de altitud:** `delta_relativo` del contrafactual (ya corregido, ver arriba) × (σ de Stuff+ / 10) = carreras por pitch que ese pitcher pierde de más o de menos que un pitch promedio en Harp Helú. **Es un cálculo del modelo, no validado contra resultados reales.**
- **Por qué Pitching+ de base** (no inventé pesos): se probó qué score de 2024 predice mejor el resultado de 2025 (181 pitchers, ≥ 50 turnos por año; K−BB% y carreras por turno ajustadas por parque): Pitching+ r = 0.36–0.38 (K−BB%) y 0.24–0.25 (carreras); Location+ 0.26 / 0.18; **Stuff+ solo 0.03–0.05 / 0.04–0.07** (Stuff+ v1; no se repitió con v3); el resultado pasado 0.43–0.47 / 0.18–0.23. (Con otra meta, Stuff+ daba r = 0.02: esa meta es débil porque `rv_tabla` valúa todo batazo en juego igual dentro de un conteo.)
- **Prueba retrospectiva** (ranking con solo 2024 → 2025, 182 pitchers con ≥ 200 pitches en 2024 —el ranking final usa ≥ 300 con las dos temporadas juntas—, sin parámetros ajustados con 2025): r = **0.38** con K−BB% (IC95 0.25–0.50; el resultado pasado 0.43) y **0.24** con carreras por turno (0.10–0.37; el resultado pasado 0.19). Los 20 primeros: K−BB% +10.0 vs +6.8 el promedio y +1.5 los 20 últimos; carreras por 100 turnos +1.2 vs −0.9 vs −3.2. **La relación es moderada: sirve para armar una lista corta, no para garantizar un nombre.** Los modelos de Pitching+ y los precios se ajustaron con 2024 y 2025 juntos, así que el modelo vio algo de 2025 de *otros* pitchers (no del pitcher evaluado); el efecto debe ser chico.
- **Resultado:** 512 pitchers con ≥ 300 pitches, **28 de los Diablos** (los que lanzan de local en Harp Helú **en la última temporada en que lanzaron ahí**; antes, el total de las dos temporadas: 33). El #1 general es uno de ellos (`pitcher_00619`, +1.66), con una advertencia: los Diablos ya traen parte de la pérdida de movimiento en su ejecución (38 % de sus pitches son en Harp Helú y Pitching+ usa el movimiento medido) y el ajuste de altitud la resta otra vez, así que su puesto no es comparable. Entre los externos: `pitcher_01110` +1.14 ± 0.20 (Sinker 42 %, Four-Seam 21 %, Slider 20 %; 808 pitches), `pitcher_00465` +1.13 ± 0.18 (1,055 pitches) y `pitcher_00458` +1.11 ± 0.21 (698 pitches) **están empatados** (diferencias de ≈ 0.03 contra incertidumbres de ≈ 0.2), seguidos de #00271 y #00019 (+1.07). Se recomienda una **lista corta** de cinco. La dispersión entre pitchers: rating sd 0.44, ejecución sd 0.41, ajuste de altitud sd 0.09 (carreras/100 pitches); es del tamaño esperado (la parte que persiste de un año a otro en los resultados reales es ≈ 0.42).
- Scores de pitcher mostrados en escala **entre pitchers** (100 + 10z, ver Paso 8): `pitching_plus`, `location_plus`, `stuff_harp`, `stuff_mar`; los promedios por pitch quedan en `*_pitch`. `stuff_harp` es el Stuff+ de Harp Helú en su propia escala.
- No considera salud, edad, contrato, rol (abridor/relevo) ni disponibilidad; los pitchers son anónimos. Salidas: `../salidas/ranking_contratacion.csv`, `prueba_retrospectiva.json`.

### Paso 11 · Productos

- Scores por lanzamiento / pitcher × tipo / pitcher; comparación por altitud.
- Perfil físico ideal para Diablos; recomendaciones operativas (quién gana/pierde en altura, cómo perfilar prospectos).
- Dashboard (Streamlit): simulador de un pitch → Stuff+; lado a lado alta vs baja; mapa de movimiento (IVB vs HB).
- Documento técnico 6–10 págs.; SHAP traducido a lenguaje de coach.

## 5. Convenciones de trabajo

- Rama por tarea (`nombre/tema`), PR con revisión, nunca push a `main`.
- `.gitignore` bloquea `*.parquet`, `*.csv` (excepto `config/*.csv`), `models/`, `modelos/`, `salidas/` y `graphify-out/` (corregido 2026-10-06: antes el archivo no tenía estas reglas aunque aquí se decía que sí).
- Semilla fija 42. Instalación con `requirements.txt`.
- Notebooks para explorar; lo definitivo va a código `.py`.

## 6. Estado (2026-10-21)

- **Hecho:** `limpieza.py` completa (bunts, `SpinAxis` y ángulos volteados, tabla de filtros), `objetivo.py`, `particion.py`, `modelo_multiclase.py` v1 (opción B + `SpinAxis`, `min_child_samples=1000`, sin altitud), `busqueda_min_child.py`, `predecir_12_conteos.py`, `run_values.py` (RV_k con corredores; corregido 2026-10-21), `scoring.py` v1 (Stuff+ neutral + contexto real), **Location+ completo** (`zona.py`, `modelo_location.py`, `busqueda_min_child_location.py`, `scoring_location.py`), **Pitching+ completo** (`pitching_plus.py`, `scoring_pitching.py`), `estadios.py`/`altitud.py` (altitud continua).
- **Altitud (Paso 10), hecho y corregido 2026-10-21:** efecto medido (`efecto_altitud.py`), `altitud_movimiento.py` (`dx/dρ = g0 + κ·x`), precio de bola en juego (`precio_in_play_altitud.py`), Stuff+ v2/v3 con arsenal (`modelo_stuff_altitud.py`, `arsenal.py`, `modelo_stuff_arsenal.py`), park holdout y contrafactual. **Stuff+ oficial = v3.**
- **Contratación (2026-10-21):** `contratacion.py` (ranking de pitchers para Harp Helú y prueba retrospectiva). **Dashboard (Paso 11):** `dashboard.py` + `simulador_altitud.py` rediseñados y probados (2026-10-21).
- **Pendiente inmediato:** `contacto.py` (Paso 7, Matías; prueba H3); H4 (eficiencia de spin); validación final (Paso 9); scores finales (reentrenar con todo); revisar la deriva de densidad de 2026 antes de usar v3 en el test sellado; pregunta a ISAC sobre altitud/estadios/clima.

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
- **2026-10-06:** **Location+ completo** (Matías). `zona.py` (`lado_relativo` + = adentro, verificado con pelotazos; zonas 1–9 + fuera, 99.999 % igual a `in_strike_zone`); `modelo_location.py` (ubicación + conteo + mano + tipo de pitch, mismos folds que Stuff+ vía `folds.csv`: con GroupKFold propio el 59 % de los pitches cambiaba de fold); `min_child_samples = 2000` por la regla pareada (1000 menor log-loss pero 2000 indistinguible, 3/5; 5000 pierde 5/5). Mejora sobre el baseline por conteo 29.18 % (ball 52.7 %, called_strike 24.3 %), calibración ≤ 1 pp por conteo. Score: `calcular_xrv(df, "real")` en `scoring.py` (xRV en el conteo real − promedio del conteo; `"neutral"` intacto, reproduce Stuff+ exacto) + `scoring_location.py`. Sanity: 3-0 dentro de la zona ≈ 114 vs fuera 89; 0-2 centro de la zona 97 y esquinas 101–102; Location+ vs tasa de bolas r = −0.74; vs Stuff+ r = 0.04; estabilidad 2024 → 2025 r = 0.62 (Stuff+ 0.78). Cambios en módulos existentes: `validacion_cruzada(..., features=None, fold_de_pitcher=None)` en `modelo_multiclase.py` y el contexto `"real"` en `scoring.py`, ambos opcionales y sin cambiar el comportamiento previo.
- **2026-10-06:** `.gitignore` corregido: no bloqueaba `*.parquet`/`*.csv` aunque §5 decía que sí (ningún dato estaba subido). Agregados `*.parquet`, `*.csv` (salvo `config/*.csv`), `models/`, `modelos/`, `salidas/`, `graphify-out/`, `.graphify/`. Guía §6: fila de `limpieza.py` al día (decía pendientes ya hechos) y convención de reusar `folds.csv`.
- **2026-10-06:** README al día (Location+, `scoring.py` ya hecho, árbol de salidas, porcentajes de clases de la re-corrida, instalación con `requirements.txt` también en Windows). Nuevo `requirements.txt` (pandas, numpy, pyarrow, lightgbm, scikit-learn, scipy; se quitó `pybaseball`, que ningún script usa): §5 ya lo daba por existente.
- **2026-10-12:** **Pitching+ modelo** (`pitching_plus.py`, Ricardo). Ve todo (8 físicas + conteo + `PlateLocHeight`/`lado_relativo` + `bateador_zurdo`/`misma_mano`), sin `tipo_pitch` ni altitud, `min_child_samples = 1000`, mismos folds que Stuff+/Location+ (`folds.csv`, `assert`). Mejora sobre el baseline por conteo **30.3 %** (Stuff+ 1.0 %, Location+ 29.2 %): casi todo es la ubicación (ball 53.6 % vs 0.02 % en Stuff+). Calibración general ≤ 0.04 pp, por conteo máx 1.13 pp (3-0). OOF con las mismas columnas que Stuff+. Nota de reparto: Pitching+ figura como de Matías en §8; lo construyó Ricardo en esta sesión — coordinar para no duplicar.
- **2026-10-12:** **Pitching+ score** (`scoring_pitching.py`, Ricardo), contexto `"real"` igual que Location+. Media 100 / sd 10 por temporada; cada conteo centrado en 0. Chequeo de la guía §4 OK: Pitching+ correlaciona con Stuff+ (r = 0.42) y con Location+ (r = 0.87), siendo Stuff+ y Location+ casi independientes (r = 0.04) → ejecución combinada dominada por la ubicación. A nivel pitcher: vs bolas r = −0.73, vs whiff r = +0.27. Estabilidad 2024→2025 r = 0.60. Salidas `scores_pitching_pitch.parquet` + agregados + `xrv_promedio_conteo_pitching.csv`.
- **2026-10-14:** **Altitud completa** (Matías). Efecto medido en la LMB por pitcher × tipo (IVB −3.7 in en la recta en Harp Helú; elasticidad ≈ 1.3, no 1.0; spin sin cambio → H1 no); `ajustar_movimiento` usa el efecto medido aditivo, **no** la proporción IVB·ρ (cambia lo que decía el Paso 10); precio de la bola en juego +0.065 carreras en Harp Helú (efecto fijo por bateador); Stuff+ **v3** (movimiento normalizado + densidad + arsenal) pasa a ser el oficial: mejora 1.25 % (v1 1.00 %), gana a v2 en 5/5 folds, corrige el sesgo de Harp Helú (97.2 → 99.1); park holdout: el método generaliza, la densidad como feature es inconclusa; contrafactual por canales (Δ promedio −26: movimiento −3.3, resultado −3.6, precio −19.2; las rectas pierden más que el promedio). Contrato cambiado: `ajustar_movimiento` vive en `altitud_movimiento.py` con `tipo` opcional (guía §6).
- **2026-10-14:** **Ranking de contratación** (Matías). `contratacion.py`: rating = ejecución (Pitching+ encogido por poca muestra) + ajuste de altitud (contrafactual), en carreras por 100 pitches; prueba retrospectiva 2024 → 2025: r = 0.39 con K−BB% y 0.25 con carreras por turno (moderada: lista corta, no certeza). **Hallazgo:** Stuff+ de un año casi no predice K−BB% ni carreras del siguiente (r ≈ 0.03–0.07), por eso el ranking usa Pitching+ de base. Cada pitcher lleva su incertidumbre, para poder decir con qué probabilidad el #1 es mejor que el #2 (66 % entre los dos primeros externos); no se da una cifra de carreras por temporada porque sobreestimaría la ventaja real.
- **2026-10-21:** **Auditoría de correctitud** (4 revisiones independientes de datos y valores de carrera, modelos y scoring, altitud, y ranking/dashboard; backup previo en `../backup_pre_auditoria_2026-10-21/`). Sin fugas de datos, signos o merges incorrectos; los scores se recalculan exactos. Hallazgos corregidos: **(crítico)** el escenario Harp Helú del contrafactual y del simulador **contaba dos veces el movimiento perdido** (el modelo se entrena con movimiento a nivel del mar + `densidad_juego`, así que la densidad ya trae el costo): el delta promedio pasa de −26.0 a −21.9, Four-Seam relativo de −5.0 a −2.4, Curveball de +8.6 a +6.6. Ajuste de movimiento: γ único por tipo → `dx/dρ = g0 + κ·x` (exponencial invertible); Sweeper usa el Slider (antes el promedio de la liga, de signo contrario). `estadios.py`: la densidad de referencia y la identificación de Harp Helú sin la temporada sellada (ρ +0.5 %, `RHO_HARP = 0.830`). `run_values.py`: precio de in_play sin bunts, walks/ponches por conteo, precio de pelotazos, HBP antes que Walk (inconsistencia máx 0.0094 → 0.0059). Ranking: confiabilidad con mitades de juegos (R(1,000) 0.88 → 0.83) y promedio previo según la muestra; Diablos por la última temporada en Harp Helú (33 → 28); correlaciones contra Stuff+ v3 (antes v1); validaciones en `scoring.py`; regla de desempate de `min_child_samples` monotónica (misma elección: 1000 / 2000). Simulador: promedio de los 5 modelos de fold (antes el modelo final, ±2.6 puntos de ruido por pitch), mano (zurdos sin voltear), recta primaria obligatoria para los que no son rectas, Stuff+ de Harp Helú en su propia escala. Dashboard: 2,232 m, mínimo 300 pitches por defecto, textos calculados en vez de fijos, barras con ratings negativos. Escala por pitcher 100 + 10z para mostrar Stuff+/Location+/Pitching+ (el orden de los pitchers no cambia). Re-corridos: `run_values`, `scoring*`, `precio_in_play_altitud`, `efecto_altitud`, `modelo_stuff_altitud`, `modelo_stuff_arsenal`, `park_holdout`, `contrafactual_altitud`, `contratacion`, `estadios`. **Revisado y NO cambiado (documentado):** reconstrucción de estadios ruidosa (`estadio_03`), deriva de densidad en 2026, `in_play` v0 sin modelo de contacto, incertidumbre del ajuste de altitud fuera de las probabilidades.
- **2026-10-21 (más tarde):** **Dashboard rediseñado desde cero** (`dashboard.py`): navegación lateral con cinco secciones, ayudas en cada número, notas de lectura, gráfica de rating con margen de incertidumbre, glosario y límites dentro del tablero, estilo sobrio sin emojis. Misma lógica de cálculo (ranking, simulador, tablas del contrafactual); probado de forma automática en todas las secciones y entradas.
