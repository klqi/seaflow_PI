"""
Run the PAR cleaning/smoothing pipeline on the full 30-minute dataset.

PAR is identical across `pop` rows, so the processing is done once per
cruise on the deduplicated (cruise, time) series. Clock corrections
detected for mislabeled cruises are applied to the whole cruise
(including the population rows), and grid-filled timestamps are
replicated for every population present on that cruise.

Outputs:
  par_processed.csv : full dataframe with norm_par_smooth, par_source, ...
  cruise_par_qc.csv : per-cruise quality table
"""

import sys
import pandas as pd

sys.path.insert(0, "/home/user/workspace/par_smoothing")
from par_quality import smooth_daily_par, flag_cruise_par_quality

INPUT = "/home/user/.perplexity/attachments/c9fd22e99af147b9b60e2cbec4b545e2/all_30min_PAR.csv"

df = pd.read_csv(INPUT, parse_dates=["time"])

# ---- 1. Deduplicated PAR series (one row per cruise + timestamp) -----------
par_cols = ["cruise", "cruise_day", "time", "time_of_day", "lat", "lon",
            "norm_par", "sst", "daylength", "correction"]
par_unique = df[par_cols].drop_duplicates(subset=["cruise", "time"]).copy()

# ---- 2. QC table (same data the processing sees) ----------------------------
qc = flag_cruise_par_quality(
    par_unique,
    cruise_col="cruise", cruise_day_col="cruise_day",
    par_col="norm_par", datetime_col="time",
    time_of_day_col="time_of_day", lat_col="lat", lon_col="lon",
)
qc.to_csv("cruise_par_qc.csv", index=False)
print(qc["flag"].value_counts().to_string())

# ---- 3. Process PAR once per cruise -----------------------------------------
processed = []
for cruise, sub in par_unique.groupby("cruise", sort=False):
    r = smooth_daily_par(
        sub,
        cruise_col="cruise", cruise_day_col="cruise_day",
        datetime_col="time", par_col="norm_par",
        time_of_day_col="time_of_day", lat_col="lat", lon_col="lon",
        min_par=5.0,
        window_length=9,   # 4.5 h; use 5 (2.5 h) to preserve more detail
        polyorder=2,
        output_col="norm_par_smooth",
    )
    processed.append(r)

processed = pd.concat(processed, ignore_index=True)

# ---- 4. Broadcast back to population rows ------------------------------------
# Apply each cruise's clock correction to the population rows too, so the
# merge keys line up.
shift_map = processed.groupby("cruise")["time_shift_h"].first().to_dict()
df_shift = df.copy()
df_shift["time"] = df["time"] - pd.to_timedelta(
    df["cruise"].map(shift_map).fillna(0.0), unit="h")

bio_cols = [c for c in df.columns if c not in par_cols and c != "pop"]
merged = processed.merge(
    df_shift[["cruise", "time", "pop"] + bio_cols],
    on=["cruise", "time"], how="left")

# Grid-filled timestamps have no biological measurements: replicate them
# for every population present on that cruise (bio columns stay NaN).
cruise_pops = df_shift.groupby("cruise")["pop"].unique().to_dict()
pops_table = pd.DataFrame(
    [(c, p) for c, ps in cruise_pops.items() for p in ps],
    columns=["cruise", "pop"])

grid_mask = merged["grid_filled"].fillna(False) & merged["pop"].isna()
grid_rows = merged[grid_mask].drop(columns=["pop"]).merge(
    pops_table, on="cruise", how="left")
par_processed = pd.concat(
    [merged[~grid_mask], grid_rows], ignore_index=True)

par_processed.to_csv("par_processed.csv", index=False)
print("saved par_processed.csv:", par_processed.shape)

# ---- 5. Quick summary ----------------------------------------------------------
print("\nRows by par_source:")
print(par_processed["par_source"].value_counts().to_string())
print("\nClock-shifted cruises:",
      {c: s for c, s in shift_map.items() if s != 0})
print("Rows with missing pop (should be 0):",
      int(par_processed["pop"].isna().sum()))
print("Night rows with nonzero smoothed PAR:",
      int((par_processed.loc[par_processed["time_of_day"] == "night",
                             "norm_par_smooth"] > 0).sum()))
