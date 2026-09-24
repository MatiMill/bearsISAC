import pandas as pd
from pybaseball import statcast

pd.set_option("display.max_columns", None)  # show every column
pd.set_option("display.width", None)        # use the full terminal width

# Download every MLB pitch from one day
df = statcast(start_dt="2025-07-01", end_dt="2025-07-01")

print(df.shape)  # (rows, columns)
print(df[["pitch_type", "pitch_name", "release_speed", "release_spin_rate",
          "pfx_x", "pfx_z", "description"]].head(10))
