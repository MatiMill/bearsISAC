# Física de la altitud · bearsISAC

**Para qué es este documento:** registrar y explicar las fórmulas de física que usa el modelo de altitud (Paso 10), con sus supuestos y límites. Sirve para que cualquiera del equipo —en especial **Fran**— pueda investigar qué tan fundadas están, verificarlas contra la literatura y aportar.

**Idea de una frase:** la altitud cambia cómo se comporta un pitch porque **el aire es menos denso**; casi todo el efecto se explica por la densidad del aire ρ, que entra en las fórmulas de arrastre y de Magnus.

---

## 1. Qué modelamos (el contrafactual)

Tomamos un pitch real y preguntamos: *¿cómo sería este mismo lanzamiento en otro estadio?* Dejamos fijo **lo que hace el brazo del pitcher** (velocidad de salida `RelSpeed`, giro `SpinRate`, eje `SpinAxis`, punto de release) y solo cambiamos **lo que el aire le hace a la bola**. Comparamos el pitch evaluado en Harp Helú (~2,232 m) vs. a nivel del mar.

El aire actúa por dos canales:
1. **Menos movimiento** (fuerza de Magnus ∝ densidad) → la bola rompe menos en altura.
2. **La bola bateada vuela más** (menos arrastre) → el contacto cuesta más carreras en altura.

---

## 2. Densidad del aire vs. altitud

### 2.1 Fórmula que usa el código (atmósfera estándar, ley de potencias)

En `altitud.py`:

$$\frac{\rho(h)}{\rho_0} = \left(1 - 2.25577\times10^{-5}\,h\right)^{4.25588}$$

- `h` = altitud en metros; `ρ₀` = densidad al nivel del mar.
- El exponente **4.25588** es el de **densidad**. (El 5.25588 sería el de **presión** — es un error común confundirlos; ver bitácora 2026-10-05.)
- Viene del modelo de **Atmósfera Estándar Internacional (ISA)**, que incluye que la temperatura baja con la altura (gradiente térmico). **A verificar por Fran:** de dónde sale el exponente y bajo qué supuestos (temperatura a nivel del mar 15 °C, gradiente 6.5 °C/km).
- En Harp Helú (2,232 m) da ρ/ρ₀ ≈ **0.80** (0.803). Con el exponente de presión (5.25588) saldría 0.76: por eso importa no confundirlos.

### 2.2 Aproximación exponencial (más simple, menos exacta)

$$\rho(h) \approx \rho_0\,e^{-h/H},\qquad H \approx 8{,}400\text{ m}$$

Es la aproximación **isotérmica** (temperatura constante). Da parecido pero no igual; **usamos la de 2.1**, que es más fiel. (Versiones viejas del documento mencionaban esta forma; quedó alineado a la ley de potencias.)

### 2.3 Lo más importante: la densidad del modelo se MIDE, no se calcula

Para el feature `densidad_juego` **no usamos estas fórmulas**: la densidad de cada juego se **infiere del frenado real de las bolas** (ver sección 4). Las fórmulas de arriba solo se usan en `altitud.py` para traducir *densidad medida ↔ altitud real en metros* (calibración de estadios). Esto es una fortaleza: la densidad medida ya incluye el efecto de temperatura y humedad de ese día.

---

## 3. Fuerza de Magnus: por qué el movimiento es ∝ densidad

El spin genera una fuerza lateral (efecto Magnus) que curva la bola:

$$F_{\text{Magnus}} = \tfrac{1}{2}\,C_L\,\rho\,A\,v^2$$

- `C_L` = coeficiente de sustentación (depende del spin y la velocidad); `A` = área transversal de la bola; `v` = velocidad; `ρ` = densidad.
- Aceleración lateral: `a = F/m` (m = masa de la bola).
- El **movimiento acumulado** (break) respecto a una bola sin spin es aproximadamente `½ · a · t²`, con `t` = tiempo de vuelo.

Como `a ∝ ρ`, manteniendo velocidad y spin fijos:

$$\boxed{\text{IVB},\,\text{HB} \;\propto\; \rho}$$

De ahí el **reescalado del contrafactual**:

$$\text{IVB}_{\text{destino}} \approx \text{IVB}_{\text{origen}}\cdot\frac{\rho_{\text{destino}}}{\rho_{\text{origen}}}\qquad(\text{igual para HB})$$

**Supuestos (y por eso es de primer orden):**
- `C_L` aproximadamente constante al cambiar la densidad (el spin no cambia con el aire; lo pone la mano).
- El tiempo de vuelo `t` aproximadamente constante — en realidad cambia un poco porque menos arrastre → la bola llega algo antes. Esto hace que la relación no sea *exactamente* lineal.
- **Por eso NO lo tomamos como dogma:** se valida empíricamente (Paso 3) con pitchers que lanzaron en densidades distintas. Si el factor real difiere del ratio, se corrige con los datos.

### 3.1 Lo que medimos en la LMB (2026-10-14)

El Paso 3 se hizo (`efecto_altitud.py`): mismo pitcher y tipo, aire distinto, 482,365 pitches y 766 pitchers, IC por bootstrap de pitchers.

- **El movimiento SÍ baja con la densidad, pero MÁS que proporcionalmente:** elasticidad β ≈ 1.3 para IVB (1.6 para HB), no 1.0. Recta Four-Seam: **−3.71 in de IVB** y −2.40 de HB en Harp Helú (ρ = 0.828) frente a nivel del mar.
  Posibles razones (sin probar): `densidad_juego` está definida con el arrastre y mezcla temperatura; el tiempo de vuelo y el coeficiente de sustentación también cambian con el aire.
- **El spin no cambia** (β = 0.00 [−0.01, 0.01]): lo pone la mano, como suponíamos (C_L constante). La velocidad de salida casi tampoco (+0.21 mph en Harp Helú). Son los controles que dicen que el efecto de IVB/HB es del aire y no de otra cosa.
- Por eso el código **no usa** el reescalado IVB·ρ_destino/ρ_origen de arriba, sino el efecto medido por tipo: `x_destino = x_origen + γ_tipo·(ρ_destino − ρ_origen)` (`altitud_movimiento.py`).
- Para Fran: pendiente explicar físicamente por qué β > 1 y por qué la curva y el slider cambian de signo en HB (menos movimiento hacia el guante).

---

## 4. Arrastre (drag) y `densidad_juego`

El arrastre también es ∝ densidad:

$$F_{\text{drag}} = \tfrac{1}{2}\,C_D\,\rho\,A\,v^2$$

Dos consecuencias:
- **Medición de la densidad:** `estadios.py` ajusta, por juego, el coeficiente de frenado de las rectas `k = a_{y0}/v^2 ∝ ρ` y define `densidad_juego = k / referencia` (referencia = mediana de los juegos "No Altitude" ⇒ ≈ 1.00 a nivel del mar, ≈ 0.82 en Harp Helú: mediana medida 0.823 en 123 juegos). La medida sale un poco más alta que la fórmula (0.80) porque la referencia no está exactamente al nivel del mar ni a 15 °C; `altitud.py` calibra esa diferencia con una recta (R² = 0.96). Como se infiere del frenado real, **captura temperatura y humedad** del día automáticamente.
- **Bola bateada:** menos arrastre en altura → la bola vuela más lejos → el contacto (`in_play`) cuesta más carreras. Ese canal lo **aprende el modelo** con `densidad_juego` como feature (y lo afinará el modelo de contacto, Paso 7).

Nota: la "velocidad que pierde la bola de camino al plato" (`ZoneSpeed`, `SpeedDrop`) **se excluye a propósito** de Stuff+ porque filtra la altitud de forma escondida; por eso ese sub-efecto de la velocidad no se modela directamente.

---

## 5. Qué NO modelamos (límites honestos)

- **Viento:** no hay datos; es direccional y cambia dentro del juego. Mete ruido (se diluye en promedio), pero si un parque tuviera viento *sistemático* sesgaría y no lo distinguiríamos.
- **Dimensiones del parque, altura del montículo, efecto sobre el bateador:** fuera del alcance; solo modelamos el canal de densidad del aire.
- **Variación bola a bola** (costuras, *seam-shifted wake*): no se modela.
- El contrafactual es de **primer orden**: aísla el efecto dominante (densidad), no es un simulador aerodinámico completo de la trayectoria.

---

## 6. Para Fran: qué investigar y verificar

Preguntas abiertas donde tu aporte (béisbol + buscar fuentes) vale mucho:

1. **¿El break realmente escala lineal con ρ?** Es nuestro supuesto central (sección 3). Lo cruzamos con los datos en el Paso 3, pero ¿qué dice la literatura de física del béisbol sobre la linealidad?
2. **Valores típicos de `C_L` y `C_D`** de una pelota de béisbol, y cómo dependen del spin y la velocidad.
3. **Cuánto más vuela una bola bateada en altura** (carry), para contrastar con lo que aprenda el modelo de contacto.
4. **Comparación con Coors Field** (Denver, ~1,580 m / 5,280 ft), el parque de altura conocido de MLB: hay mucho escrito sobre su efecto. **Harp Helú (~2,232 m) es aún más alto**, así que el efecto debería ser mayor — ¿cuánto dicen las fuentes?

Fuentes para empezar a investigar (verificar, no citamos nada aún):
- Robert K. Adair, *The Physics of Baseball* (libro clásico).
- Alan M. Nathan — físico con trabajos y recursos públicos sobre aerodinámica de la pelota (drag/lift, Magnus).
- El modelo de **Atmósfera Estándar Internacional (ISA)** para la fórmula de densidad de la sección 2.
- Artículos de Stuff+ (p. ej. de FanGraphs / Baseball Prospectus / Driveline) para ver cómo otros tratan el movimiento y la altitud.

Cuando encuentres algo que confirme, matice o contradiga estos supuestos, anótalo aquí y avísanos para actualizar `decisions.md`.

---

## 7. Dónde vive cada cosa en el código

| Fórmula | Archivo | Uso |
|---|---|---|
| Densidad vs. altitud (sección 2.1) | `altitud.py` (`densidad_esperada`) | Traducir densidad medida ↔ altitud real de los estadios |
| `densidad_juego` medida del frenado (sección 4) | `estadios.py` | Feature continuo de altitud por juego |
| Reescalado IVB/HB ∝ ρ (sección 3) | `scoring.py` / módulo de altitud (Paso 10, pendiente) | Contrafactual Harp Helú vs. nivel del mar |
