# Guía del proyecto · bearsISAC (Stuff+ LMB)

**Para quién es:** para cualquier persona del equipo y para su asistente de IA (Claude Code u otro).
**Para qué sirve:** que todos entiendan lo mismo sobre qué construimos, cómo funciona cada pieza y qué entrega cada módulo al siguiente, aunque trabajen por separado.

**Cómo usar este documento (instrucciones para el asistente):**
1. Lee este archivo completo y después `docs/decisions.md`. Este archivo explica el *qué* y el *por qué*; `decisions.md` es el registro detallado de decisiones y resultados verificados.
2. Si algo aquí contradice `decisions.md`, manda `decisions.md` (es la fuente de verdad) y avisa de la contradicción.
3. Respeta los **contratos** de la sección 6: nombres de funciones, archivos, columnas y convenciones. Es lo que permite que el trabajo de distintas personas encaje.
4. No cambies una decisión ni un contrato por tu cuenta. Si crees que algo debería cambiar, explícalo a la persona con la que trabajas; si lo aprueba, actualiza `docs/decisions.md` (sección correspondiente y bitácora con fecha) y avísale que debe comunicarlo al equipo.
5. Explica en español, paso a paso y con el porqué de cada decisión.

---

## 1. El reto en una frase

Construir un **Stuff+ calibrado para la Liga Mexicana de Béisbol (LMB)**: un número que diga qué tan bueno es cada lanzamiento por su **calidad esperada** (no por lo que le pasó), y cuantificar cuánto cambia esa calidad en el **Estadio Harp Helú** (Diablos Rojos, ~2,232 m) frente a estadios a nivel del mar.

- **Cliente final:** el cuerpo técnico de Diablos Rojos (pitching coach y director deportivo).
- **Entrega:** 2026-10-15 (meta interna: 14 de octubre). Finalistas el 20 de octubre; presentaciones el 10–11 de noviembre.
- **Datos:** 3 temporadas pitch-by-pitch de la LMB (`stuff_model_df.parquet`, ~628 mil lanzamientos después de limpiar). Solo pueden usarse para la competencia y **nunca se suben a git**.

### Entregables obligatorios
1. Scores Stuff+, Location+ y Pitching+ por lanzamiento, por pitch de cada pitcher y por pitcher; comparación CDMX vs. otros estadios.
2. Perfil físico ideal de pitcher para Diablos (qué características maximizan Stuff+ en Harp Helú).
3. Recomendaciones operativas: qué pitches ganan y cuáles pierden en altitud, y cómo perfilar prospectos.
4. Dashboard simulador: el coach mete la física de un pitch y ve su Stuff+ en Harp Helú vs. nivel del mar, más un mapa de movimiento.
5. Documento técnico de 6–10 páginas.

### Cómo nos evalúan (25% cada uno)
- **Calidad del modelo:** bien calibrado y mejor que el baseline (xRV promedio por conteo).
- **Robustez fuera de muestra:** funciona con pitchers no vistos (pitcher holdout) y con estadios distintos (park holdout).
- **Integración de arsenal y altitud:** features relacionales de arsenal; efecto de altitud cuantificado con datos de la LMB.
- **Aplicabilidad real:** entrevista de 10 minutos con el cuerpo técnico.

---

## 2. Béisbol mínimo necesario

- Un **turno** (plate appearance) es el duelo de un bateador contra el pitcher; termina en out, hit, base por bolas, ponche o pelotazo.
- **Conteo** = bolas-strikes antes del lanzamiento (12 posibles: 0-0 … 3-2). 4 bolas = base por bolas; 3 strikes = ponche; un foul cuenta como strike salvo con 2 strikes.
- Una **entrada** termina con 3 outs. Las **carreras** son lo que gana partidos.
- Cada lanzamiento termina en uno de **5 resultados**: ball, called strike (strike cantado sin swing), whiff (swing y falla), foul, in play (la mete al campo).
- La física del pitch: velocidad (`RelSpeed`), spin (`SpinRate`, `SpinAxis`), movimiento por spin (`InducedVertBreak` = IVB, `HorzBreak` = HB), punto de salida (`RelHeight`, `RelSide`, `Extension`).
- **Altitud:** el aire menos denso reduce el movimiento por spin (efecto Magnus, proporcional a la densidad) y hace que la bola bateada vuele más. En Harp Helú la densidad es ~80% de la del nivel del mar.

---

## 3. La idea central de Stuff+

### 3.1 Se aprende con resultados, se califica sin ellos

- **Entrenar (una vez):** con cientos de miles de lanzamientos y **lo que les pasó**, un modelo aprende qué suele pasar con pitches de cada física.
- **Calificar (cada pitch):** solo entra la **física** del pitch. Su valor viene de lo que el modelo aprendió de **todos los demás** pitches parecidos.
- Consecuencia: dos pitches con la misma física tienen el mismo Stuff+, aunque uno haya terminado en ponche y el otro en jonrón. El resultado de un pitch individual es sobre todo suerte; el promedio de miles no.
- No hay reglas escritas a mano ("más velocidad = mejor"). Casi ninguna regla simple es cierta en general: mucho IVB es bueno en una recta four-seam y malo en un sinker; poco spin es bueno en un cambio. El modelo descubre estas interacciones solo.

### 3.2 La fórmula central: valor esperado

$$\text{xRV} = \sum_{k \in \{\text{ball, called, whiff, foul, in play}\}} P(k \mid \text{pitch}) \cdot RV_k(\text{conteo})$$

- $P(k \mid \text{pitch})$: probabilidad de cada resultado. **La da el modelo** (LightGBM multiclase).
- $RV_k(\text{conteo})$: **precio** de cada resultado en carreras, según el conteo. **Sale de contar en los datos** (sección 5). No es un modelo.
- xRV: cuántas carreras vale el pitch en promedio. **Convención de signo: positivo = bueno para el bateador.**

### 3.3 De xRV a Stuff+

1. $s = -\text{xRV}$ (más alto = mejor para el pitcher).
2. $z = (s - \mu)/\sigma$, con $\mu$ y $\sigma$ de **todos los lanzamientos de la liga en esa temporada**.
3. $\text{Stuff+} = 100 + 10z$. 100 = promedio; 110 = una desviación estándar mejor; 120+ = élite.
4. Agregación: lanzamiento → promedio por pitcher × tipo de pitch → promedio del pitcher ponderado por uso.

---

## 4. Arquitectura: tres modelos separados

| Modelo | Pregunta | Qué ve el modelo | En qué contexto se califica |
|---|---|---|---|
| **Stuff+** | ¿Qué tan buena es la bola? | Física + arsenal + densidad del aire + `Balls`, `Strikes` | **Neutral:** promedio sobre los 12 conteos (pesos de la liga) y en aire promedio de la liga |
| **Location+** | ¿La puso donde debía? | Ubicación (`PlateLocHeight`, `PlateLocSide`) + conteo + lado del bateador + tipo de pitch | Conteo real, restando el xRV promedio de ese conteo |
| **Pitching+** | ¿Qué tan bueno fue este lanzamiento concreto? | Todo lo anterior | Conteo real (restando el promedio del conteo) y altitud real |

- **Pitching+ se entrena desde cero** con todas las columnas. **No** es promedio de Stuff+ y Location+: el valor de la ubicación depende del pitch (una recta con mucho IVB arriba en la zona es buena; un sinker arriba es malo), y las escalas no se pueden sumar.
- Los tres usan el mismo método (LightGBM multiclase → xRV → escala 100 + 10z). Solo cambian las columnas y el contexto de evaluación.
- **Nunca entran como features:** IDs (`pitcher_anon_id`, etc.: el modelo memorizaría pitchers), resultados (`PitchCall`, `play_result`…), y en Stuff+ tampoco la ubicación. También se evitan en Stuff+ `ZoneSpeed`, `SpeedDrop` y `ZoneTime` (dependen de la densidad del aire) y los ángulos de llegada crudos (dependen de la ubicación).
- **Regla de oro de features:** algo que se sabe en el momento en que la bola sale de la mano.

### 4.1 El conteo: "opción B"

- `Balls` y `Strikes` **entran** como features del modelo de Stuff+.
- Para calificar, el mismo pitch se evalúa **en los 12 conteos** y se promedia con los pesos de la liga (qué fracción de lanzamientos ocurre en cada conteo):
  $$\text{xRV}_{\text{Stuff}} = \sum_{c=1}^{12} w_c \cdot \text{xRV}(\text{física}, c)$$
- **Por qué:** los pitches se usan en conteos distintos (sliders en 0-2, rectas en 3-0). Sin el conteo, el modelo atribuiría al pitch lo que en realidad fue del conteo (sesgo de selección).
- Combinaciones raras (curva en 3-0): el modelo combina el efecto del conteo y el de la física; los conteos raros pesan poco (3-0 ≈ 2%). Vigilar la tabla tipo de pitch × conteo y comparar con la versión sin conteo (correlación > 0.9).

### 4.2 Ejemplo completo (números ilustrativos)

Slider: 85 mph, 2,500 rpm, IVB 1", HB −7" (hacia el guante), lanzado en Harp Helú en 1-2.

| Conteo | P(ball) | P(called) | P(whiff) | P(foul) | P(in play) | xRV(c) |
|---|---|---|---|---|---|---|
| 0-0 | 0.40 | 0.18 | 0.14 | 0.13 | 0.15 | +0.0025 |
| 1-2 | 0.38 | 0.05 | 0.22 | 0.19 | 0.16 | −0.0276 |
| 3-1 | 0.45 | 0.20 | 0.08 | 0.12 | 0.15 | +0.0345 |

Promedio ponderado de los 12 conteos: xRV = −0.006 → s = +0.006 → con μ = 0, σ = 0.004: z = 1.5 → **Stuff+ = 115**. Nunca entraron: lo que le pasó, su conteo real, su ubicación ni quién lo lanzó.

---

## 5. Los precios RV_k: de dónde salen

**Qué son:** una tabla de 12 conteos × 5 resultados con el valor en carreras de cada resultado. Ejemplo: un whiff en 1-2 (tercer strike = ponche) vale mucho más para el pitcher que un strike en 0-0.

**Idea:** precio de un resultado = (qué tan bien le va al bateador desde el conteo al que pasa) − (qué tan bien le iba desde el conteo donde estaba).

**Cálculo (solo promedios y restas, con datos SIN limpiar de las temporadas de entrenamiento):**

El dataset no dice qué corredores hay en base, pero las filas vienen en **orden cronológico**, así que se puede contar **cuántos**: `corredores = bateadores que terminaron su turno − outs − carreras`. El estado de la entrada es (outs, corredores): 12 estados (3 niveles de outs × 4 de corredores, 0–3).

> **Por qué contar corredores (corrección 2026-10-05):** medir RE solo por outs falla — una base por bolas salía valiendo +0.03 carreras (MLB ≈ +0.3), porque el corredor que llega a base no cambia los outs, y entonces las bolas en 3-0 salían negativas. Como los outs no decrecen dentro de la media entrada (99.8 %), el número de corredores reconstruido es confiable y arregla el problema.

1. **RE(outs, corredores):** carreras promedio desde ese estado hasta que termina la media entrada. RE = 0 cuando la entrada se acaba.
2. **Valor de cada final de turno** (lanzamiento con `KorBB` ∈ {Strikeout, Walk} o `PitchCall` ∈ {InPlay, HitByPitch}):
   valor = RE(estado al terminar) − RE(estado al empezar) + `RunsScored` del turno.
   Promediado por tipo de evento da los *linear weights*. **Resultado 2024–2025:** BB +0.42, 1B +0.54, 2B +0.69, 3B +0.84, HR +1.44, K −0.34, out en juego −0.36 (cerca de MLB).
3. **V(conteo):** promedio del valor final de los turnos que pasaron por ese conteo. Turno = media entrada + `batter_anon_id`; usar pares únicos (turno, conteo) porque los fouls con 2 strikes repiten conteo; descartar turnos con más de un final.
4. **RV_k(b, s):**
   - ball: V(b+1, s) − V(b, s); con 3 bolas: V_BB − V(3, s).
   - called strike y whiff: V(b, s+1) − V(b, s); con 2 strikes: V_K − V(b, 2).
   - foul: igual que strike si s < 2; **0** si s = 2.
   - in play (v0): promedio del valor de los InPlay desde ese conteo − V(b, s). En v1 lo reemplaza el modelo de contacto.

Pasa 7 revisiones: RE baja con outs y sube con corredores, V sube con bolas y baja con strikes, toda bola > 0, todo strike < 0, los pesos suman 1.

**Aproximaciones a documentar:** se sabe **cuántos** corredores hay, no en qué base (RE promedia sobre las configuraciones de base con el mismo número de corredores); HitByPitch se trata como ball; carreras en lanzamientos que no terminan el turno entran en RE pero no se atribuyen al turno.

---

## 6. Contratos entre módulos

Cada módulo lee y escribe cosas con estos nombres. Si respetas el contrato, puedes cambiar todo lo de adentro sin romper el trabajo de otros. Las salidas con datos van **fuera del repo**, junto al parquet (`../`).

| # | Módulo | Recibe | Entrega | Estado |
|---|---|---|---|---|
| 1 | `limpieza.py` | `../stuff_model_df.parquet` | `cargar_datos() -> DataFrame` limpio, con zurdos volteados (`HorzBreak`, `RelSide`, `HorzRelAngle`, `HorzApprAngle` ×(−1); `SpinAxis` → 360 − `SpinAxis`), sin bunts; tabla de filas quitadas | Hecho |
| 2 | `objetivo.py` | DataFrame | `CLASES = ["ball","called_strike","whiff","foul","in_play"]`; `agregar_resultado(df)` agrega `outcome_class` y `outcome_id` (0–4, en ese orden) | Hecho y verificado |
| — | `estadios.py` | parquet crudo | `../estadios.parquet`: `game_anon_id, year, estadio, altitud, densidad_juego, harp_helu` | Hecho (Matías) |
| — | `altitud.py` | `../estadios.parquet` | `../altitudes.parquet`: por estadio, densidad medida, altitud real y estimada | Hecho (exponente de densidad 4.25588, R² 0.96) |
| 4 | `particion.py` | DataFrame | `separar_test(df) -> (entrenamiento, test)`; test = última temporada (`year`) | Hecho |
| 5 | `run_values.py` (nuevo) | parquet crudo (sin limpiar), solo temporadas de entrenamiento | `../salidas/rv_tabla.csv`: `balls, strikes, outcome_class, rv`; `../salidas/valores_evento.csv` (linear weights); `../salidas/pesos_conteo.csv`: `balls, strikes, peso` | Pendiente |
| 6 | `modelo_multiclase.py` | entrenamiento (v2: + `estadios.parquet`) | `../salidas/predicciones_oof.parquet`: `PitchUID, pitcher_anon_id, Balls, Strikes, outcome_class, p_ball, p_called_strike, p_whiff, p_foul, p_in_play`; `../modelos/modelo_resultado.txt` + `modelo_resultado_fold{1..5}.txt`; `../salidas/folds.csv` (`pitcher_anon_id, fold`); tabla de validación | v1 hecho (sin altitud, `min_child_samples = 1000`); v2 agrega `densidad_juego` |
| 6/8 | `predecir_12_conteos.py` | entrenamiento + `folds.csv` + modelos de fold | `../salidas/predicciones_12conteos.parquet`: `PitchUID, pitcher_anon_id, Balls, Strikes, p_ball, p_called_strike, p_whiff, p_foul, p_in_play` (largo, `p_*` float32; una fila por pitch × 12 conteos; los demás datos se unen por `PitchUID`) | Hecho (v1, out-of-fold) |
| L | `zona.py` | DataFrame limpio | `preparar_ubicacion(df)`: quita `BatterSide` desconocido y agrega `lado_relativo` (`PlateLocSide` con signo del bateador, + = adentro), `zona` (1–9 dentro de la zona de strike, 10 fuera), `bateador_zurdo`, `misma_mano`, `tipo_pitch` (categórica con `TIPOS_VALIDOS`) | Hecho (Matías) |
| L | `modelo_location.py` | entrenamiento + `folds.csv` | `FEATURES_LOCATION`; `../salidas/predicciones_oof_location.parquet` (mismas columnas que `predicciones_oof.parquet`); `../salidas/validacion_location.csv`; `../modelos/modelo_location_fold{1..5}.txt` + `modelo_location.txt` | Hecho (Matías) |
| L | `scoring_location.py` | `predicciones_oof_location.parquet` + `rv_tabla.csv` | `../salidas/scores_location_pitch.parquet` (`PitchUID, pitcher_anon_id, AutoPitchType, year, Balls, Strikes, xrv_location, xrv_vs_conteo, location_plus`), `scores_location_pitcher_tipo.csv`, `scores_location_pitcher.csv`, `xrv_promedio_conteo_location.csv` (`Balls, Strikes, xrv_conteo`) | Hecho (Matías) |
| 7 | `contacto.py` (nuevo) | lanzamientos in play | modelo de E[valor | contacto, pitch, densidad]; reemplaza RV de in play | Pendiente |
| 8 | `scoring.py` | `predicciones_12conteos.parquet` + `rv_tabla.csv` + `pesos_conteo.csv` | `calcular_xrv(df, contexto)` con contexto `"neutral"` (Stuff+) o `"real"` (Location+/Pitching+); `a_escala_plus(s, year)`; `../salidas/scores_pitch.parquet`, `scores_pitcher_tipo.csv`, `scores_pitcher.csv`. Contexto `"real"`: una fila por pitch en su conteo real → `PitchUID, pitcher_anon_id, Balls, Strikes, xrv, xrv_conteo, xrv_vs_conteo` (`promedio_conteo` opcional para pitches nuevos) | Hecho (v1: Stuff+ neutral y contexto real; falta altitud) |
| 10 | altitud (en `scoring.py`) | pitch + densidades | `ajustar_movimiento(ivb, hb, rho_origen, rho_destino)` y delta de Stuff+ Harp Helú vs. nivel del mar | Pendiente |
| 11 | dashboard | `scoring.py` + modelos | Streamlit: simulador, lado a lado, mapa de movimiento | Pendiente |

### Convenciones que todos respetan
- **Signo:** RV y xRV positivos = buenos para el bateador. Stuff+/Location+/Pitching+ más altos = mejores para el pitcher.
- **Orden de clases:** el de `CLASES` en `objetivo.py`. Las columnas de probabilidad se llaman `p_<clase>`.
- **Zurdos:** después de `limpieza.py`, HB negativo = hacia el guante, sin importar la mano.
- **Altitud:** variable `densidad_juego` (1.00 = mediana de juegos "No Altitude"); Harp Helú identificado por `harp_helu`.
- **Validación:** siempre agrupada por `pitcher_anon_id` (GroupKFold). Nunca splits aleatorios por lanzamiento. Los modelos que no son Stuff+ (Location+, Pitching+) reusan **la misma partición** leyendo `../salidas/folds.csv` (`validacion_cruzada(..., fold_de_pitcher=...)`), para que las predicciones out-of-fold de los tres modelos sean comparables pitch a pitch.
- **Test sellado:** la última temporada no se usa para entrenar, elegir hiperparámetros ni calcular RV_k. Solo para la validación final.
- **Semilla:** 42.
- **Rutas:** relativas al archivo (`Path(__file__).resolve().parent.parent`), nunca rutas absolutas de una computadora.

---

## 7. El pipeline completo, en orden

1. **Limpiar** (`limpieza.py`): quitar lo que no es competencia real (intencionales, automáticas, bunts), lo que no se puede describir y errores de medición; voltear zurdos.
2. **Etiquetar** (`objetivo.py`): 5 clases desde `PitchCall`. Verificado: ball 37.8%, called 15.4%, whiff 11.3%, foul 17.2%, in play 18.3%; whiff coincide 100% con `is_whiff`; 1,119 filas (0.18%) con `is_swing = 1` pero ball/called strike (probables swings a medias); se mantiene la clase según `PitchCall`.
3. **Features:** física; arsenal por pitcher × temporada (recta primaria = tipo más usado entre Four-Seam/Sinker/Cutter; diferencias de velocidad, IVB, HB y release contra ella; uso); `densidad_juego`; `Balls`, `Strikes`.
4. **Separar** el test sellado (última temporada).
5. **Valores de carrera** (sección 5).
6. **Modelo de resultado:** LightGBM multiclase, log-loss, GroupKFold(5) por pitcher, early stopping con 10% de pitchers apartados, predicciones out-of-fold. Baseline: frecuencia de cada clase dentro de cada conteo. `min_child_samples` se elige probando {20, 50, 100, 200, 500, 1000, 2000} con la misma validación: menor log-loss; si la diferencia **pareada fold a fold** con otro valor más grande no es consistente (ese valor menor no gana en al menos 4 de 5 folds), preferir el más grande. **No** usar la desviación de la log-loss entre folds como tolerancia de empate: está dominada por la distinta dificultad de cada fold (común a todos los valores) y la comparación pareada la cancela. Justificación teórica: error estándar de una proporción √(p(1−p)/n). Elegido 2026-10-05: `min_child_samples = 1000` (ver `decisions.md`).
7. **Modelo de contacto:** solo in play; valor del contacto según el pitch y la densidad (en altura los elevados cuestan más).
8. **xRV y escala:** sección 3 y 4.
9. **Validación final** en el test sellado: contra baseline, calibración, park holdout (entrenar sin Harp Helú o sin altitud extrema y probar ahí).
10. **Altitud:** delta de Stuff+ del mismo pitch en Harp Helú vs. nivel del mar, por tipo de pitch. Dos canales: (1) menos movimiento — ya viene en IVB/HB medidos, así que para el contrafactual se ajusta IVB_destino ≈ IVB_origen · ρ_destino/ρ_origen (igual HB); (2) contacto más caro — lo aprende el modelo vía la densidad. Las **fórmulas de física y sus supuestos** (densidad vs. altitud, Magnus, arrastre, reescalado) están en [`docs/fisica_altitud.md`](fisica_altitud.md). Hipótesis del brief: H1 (spin pierde en altura), H2 (cambio gana relativo), H3 (groundball pitchers valen más), H4 (spin rate solo no basta; usar eficiencia de spin con `ax0`, `az0`).
11. **Productos:** scores, perfil ideal, recomendaciones, dashboard, documento técnico.

---

## 8. Cómo trabajamos en paralelo sin desalinearnos

- **Una sola fuente de verdad:** `docs/decisions.md`. Todo resultado verificado y toda decisión nueva se anota ahí, con línea en la bitácora (fecha y motivo).
- **Cada quien en su rama** (`nombre/tema`), pull request para juntar, `git pull` antes de empezar.
- **No edites el módulo de otra persona sin avisar.** Si necesitas algo de un módulo ajeno, pídelo respetando su contrato.
- **Al terminar un paso:** correr, verificar, anotar resultados en `decisions.md`, commit y PR.
- **Si un contrato debe cambiar** (nombre de columna, archivo de salida, firma de función): primero se acuerda con el equipo, luego se actualiza la sección 6 de este archivo y `decisions.md`.
- **Datos:** nunca hacer `git add` de `.parquet` ni `.csv` (salvo `config/*.csv`); revisar `git status` antes de cada commit. No imprimir filas individuales de datos si no hace falta: preferir conteos, porcentajes y métricas.

### Reparto (actualizado 2026-10-05)
| Persona | Pasos |
|---|---|
| Ricardo | 4 (partición), 6 para **Stuff+**: modelo de probabilidades P(k \| física, conteo, densidad), optimización de `min_child_samples` y predicción en los 12 conteos; 8 (scoring y escala de Stuff+) |
| Matías | **Location+ y Pitching+** (sus modelos y su scoring, reutilizando `objetivo.py`, la partición, `rv_tabla.csv` y el mismo esquema de validación); correcciones de `altitud.py` y `limpieza.py`; 7 (contacto); 10 (contrafactual de altitud); 11 (dashboard) |
| Ivana | 5 (valores de carrera), validación y calibración (9), física de la altitud |
| Fran | Revisión con ojo de béisbol, perfil ideal, recomendaciones, documento técnico, presentación |

---

## 9. Preguntas abiertas
- ¿Qué tan claro sale Harp Helú frente al segundo mejor local de altitud extrema? ¿Qué R² tiene la recta de `altitud.py`?
- ✅ Resuelta: las 1,119 filas de swing (`is_swing = 1` pero `PitchCall` de pitch tomado) quedaron documentadas en el Paso 2 de `decisions.md` (BallCalled 457, StrikeCalled 390, HitByPitch 227, BallinDirt 45; se mantiene la clase según `PitchCall`).
- ¿Usamos `densidad_juego` (continua) o el estadio como variable de altitud en el modelo? Propuesta: densidad continua.
