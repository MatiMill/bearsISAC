import pandas as pd
from pybaseball import statcast

pd.set_option("display.max_columns", None)  # show every column
pd.set_option("display.width", None)        # use the full terminal width

# Download every MLB pitch from one day
df = statcast(start_dt="2025-07-01", end_dt="2025-07-01")

print(df.shape)  # (rows, columns)
#cambia head para cambiar numero de resultados 
print(df[["pitch_type", "pitch_name", "release_speed", "release_spin_rate",
          "pfx_x", "pfx_z", "description"]].head(5))

summary = df.groupby("pitch_name").agg(
    #2nd argument identifies what to calculate and display
    count=("pitch_name", "size"),
    avg_speed=("release_speed", "mean"),
    # agregado mio para checar funcionamiento fastestpitch=("release_speed", "max"),
    avg_spin=("release_spin_rate", "mean"),
    avg_pfx_x=("pfx_x", "mean"),
    avg_pfx_z=("pfx_z", "mean"),
).sort_values("count", ascending=False)

print("------------------------------------------")

print(summary.round(2))

