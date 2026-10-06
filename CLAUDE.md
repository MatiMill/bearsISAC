# bearsISAC · Stuff+ para la LMB (Hackathon ISAC 2026)

Modelo Stuff+ / Location+ / Pitching+ para la Liga Mexicana de Béisbol, calibrado por altitud
(Diablos Rojos, Estadio Harp Helú, ~2,232 m). Entrega: 2026-10-15.

## Antes de cualquier tarea
- Lee `docs/guia_proyecto.md` (qué construimos, cómo funciona cada pieza y los contratos entre módulos)
  y después `docs/decisions.md`: es la fuente de verdad de todas las decisiones de modelado
  (tres modelos separados, conteo con opción B, altitud, valores de carrera, validación).
- Respeta los contratos de la sección 6 de la guía (nombres de funciones, archivos y columnas).
- No contradigas una decisión de ese archivo sin decirlo explícitamente. Si una decisión
  cambia, actualiza `docs/decisions.md` (sección y bitácora con fecha).

## Datos (reglas estrictas)
- Los datos del ISAC solo pueden usarse para la competencia y **nunca se suben a git**.
- `stuff_model_df.parquet` vive una carpeta arriba del repo (`../stuff_model_df.parquet`).
  Las salidas derivadas (`estadios.parquet`, `altitudes.parquet`, `salidas/`, `modelos/`)
  también van fuera del repo.
- Nunca hagas `git add` de `.parquet` ni `.csv` (excepto `config/*.csv`). Revisa `git status` antes de cada commit.
- No imprimas filas individuales de los datos en la conversación si no hace falta; prefiere
  conteos, porcentajes y resúmenes.

## Cómo trabajar con el equipo
- Equipo: Ricardo, Matías, Ivana y Fran. Cada quien trabaja en su rama y con su propio asistente;
  la coordinación pasa por `docs/decisions.md` y la guía.
- Explica en español, paso a paso, con el porqué de cada decisión y ejemplos con números.
  Ricardo está aprendiendo béisbol y machine learning (nivel de mates: IB AA HL).
- Avanza un paso a la vez y confirma antes de pasar al siguiente.
- Entornos: Ricardo usa Windows + PowerShell (`.venv\Scripts\Activate.ps1`); Matías usa WSL/Linux
  (`source ~/venvs/bears/bin/activate`). El código debe funcionar en ambos (rutas con `pathlib`).

## Archivos
| Archivo | Qué hace |
|---|---|
| `limpieza.py` | Carga y limpia los datos; voltea zurdos |
| `objetivo.py` | Paso 2: target de 5 clases (ball, called_strike, whiff, foul, in_play) |
| `modelo_multiclase.py` | Paso 6: LightGBM multiclase con GroupKFold por pitcher |
| `modelo.py` | Versión inicial: P(whiff | swing). Solo referencia |
| `estadios.py` | Reconstruye equipos/estadios y densidad del aire por juego (Matías) |
| `altitud.py` | Calibra densidad medida vs. altitud real por estadio (Matías) |
| `explorar_columnas.py` | Inventario de columnas del dataset |
| `docs/guia_proyecto.md` | Qué construimos, cómo funciona y contratos entre módulos |
| `docs/decisions.md` | Decisiones, resultados verificados y bitácora |

## Git
- Rama por tarea (`nombre/tema`), pull request con revisión, nunca push directo a `main`.
- Hacer `git pull` antes de empezar.
