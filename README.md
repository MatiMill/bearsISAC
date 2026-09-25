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
└── bearsISAC/                        ← este repo
    ├── limpieza.py
    └── modelo.py
```

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
```bash
python modelo.py
```

## Archivos
| Archivo | Qué hace |
|---|---|
| `limpieza.py` | Carga los datos y los limpia: quita pitches no competitivos, filas vacías y valores imposibles; voltea el movimiento horizontal de los zurdos. |
| `modelo.py` | Entrena un modelo LightGBM que predice la probabilidad de swing fallado (`is_whiff`), lo evalúa con pitchers que no vio en el entrenamiento y calcula el Stuff+. |
| `prueba_baseball.py` | Práctica con datos públicos de MLB (pybaseball). |

## Cómo trabajamos con Git
- Cada quien trabaja en su propia rama: `nombre/tema`.
- Nunca hacer push directo a `main`: usar un pull request.
- Hacer `git pull` antes de empezar a trabajar.
- Nunca subir archivos de datos (`.parquet`, `.csv`).