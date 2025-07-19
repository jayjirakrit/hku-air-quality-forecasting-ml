import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import ast


# Load csv
def generate_image(base_dir):
    stations_epd = pd.read_csv(base_dir + "stations_epd_idx.csv")
    stations_hko = pd.read_csv(base_dir + "stations_hko_idx.csv")

    dfs = {}

    dfs["env"] = pd.read_csv(base_dir + "air_quality_env.csv")
    dfs["idx"] = pd.read_csv(base_dir + "air_quality_idx.csv")
    dfs["hum"] = pd.read_csv(base_dir + "humidity.csv")
    dfs["tmp"] = pd.read_csv(base_dir + "temperature.csv")
    dfs["prs"] = pd.read_csv(base_dir + "pressure.csv")
    dfs["wnd"] = pd.read_csv(base_dir + "wind.csv")

    # Rename columns & parse datetimes
    dfs["env"].rename(columns={"report_datetime": "datetime"}, inplace=True)
    dfs["idx"].rename(columns={"report_datetime": "datetime", "agi": "aqi"}, inplace=True)
    dfs["hum"].rename(
        columns={"Hour": "datetime", "Relative Humidity(percent)": "humidity", "Station": "station"}, inplace=True
    )
    dfs["tmp"].rename(
        columns={
            "Hour": "datetime",
            "Average Max Temp": "max_temp",
            "Average Min Temp": "min_temp",
            "Station": "station",
        },
        inplace=True,
    )
    dfs["prs"].rename(
        columns={
            "Datetime": "datetime",
            "Mean Sea Level Pressure(hPa)": "pressure",
            "Automatic Weather Station": "station",
        },
        inplace=True,
    )
    dfs["wnd"].rename(
        columns={
            "Datetime": "datetime",
            "10-Minute Mean Wind Direction(Compass points)": "wind_direction",
            "10-Minute Mean Speed(km/hour)": "wind_speed",
            "10-Minute Maximum Gust(km/hour)": "max_wind_speed",
            "Automatic Weather Station": "station",
        },
        inplace=True,
    )

    for k, df in dfs.items():
        df["datetime"] = pd.to_datetime(df["datetime"])
        dfs[k] = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Determine the 48-hour window
    # take the max across all dfs to be safe
    max_ts = max(df["datetime"].max() for df in dfs.values())
    start = max_ts - timedelta(hours=47)
    time_full_idx = pd.date_range(start, max_ts, freq="1H")
    print(f"Selecting time from {time_full_idx[0]} to {time_full_idx[-1]}")
    def format_wind_direction(wind_direction):
        bearing = {
            "North": 0,
            "Northeast": 45,
            "East": 90,
            "Southeast": 135,
            "South": 180,
            "Southwest": 225,
            "West": 270,
            "Northwest": 315,
        }

        if type(wind_direction) == float:
            wind_direction = None
        elif "[" in wind_direction:
            wind_direction = set(ast.literal_eval(wind_direction.replace(" ", ","))) - set(["Calm", "Variable", "nan"])
        else:
            wind_direction = set([wind_direction]) - set(["Calm", "Variable", "nan"])

        if wind_direction == set():
            cardinal = -10
        elif wind_direction is None:
            cardinal = None
        else:
            cardinal = sum([bearing[d] for d in wind_direction])

        return cardinal

    dfs["wnd"]["wind_direction"] = dfs["wnd"]["wind_direction"].apply(format_wind_direction)

    def temporal_interpolate_fixed(df, station_col, time_col, value_cols, full_idx):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        out = []
        for stn, grp in df.groupby(station_col, sort=False):
            grp = grp.set_index(time_col).sort_index()
            grp = grp.reindex(full_idx)  # <-- only 48 timestamps
            grp[value_cols] = grp[value_cols].interpolate(method="time", limit_direction="both")
            grp[station_col] = stn
            out.append(grp)
        result = pd.concat(out, axis=0).reset_index().rename(columns={"index": time_col})
        return result

    dfs["env"] = temporal_interpolate_fixed(
        dfs["env"],
        station_col="station",
        time_col="datetime",
        value_cols=["so2", "no", "no2", "rsp", "o3", "fsp"],
        full_idx=time_full_idx,
    )
    dfs["idx"] = temporal_interpolate_fixed(
        dfs["idx"], station_col="station", time_col="datetime", value_cols=["aqi"], full_idx=time_full_idx
    )
    dfs["hum"] = temporal_interpolate_fixed(
        dfs["hum"], station_col="station", time_col="datetime", value_cols=["humidity"], full_idx=time_full_idx
    )
    dfs["tmp"] = temporal_interpolate_fixed(
        dfs["tmp"],
        station_col="station",
        time_col="datetime",
        value_cols=["max_temp", "min_temp"],
        full_idx=time_full_idx,
    )
    dfs["prs"] = temporal_interpolate_fixed(
        dfs["prs"], station_col="station", time_col="datetime", value_cols=["pressure"], full_idx=time_full_idx
    )
    dfs["wnd"] = temporal_interpolate_fixed(
        dfs["wnd"],
        station_col="station",
        time_col="datetime",
        value_cols=["wind_speed", "max_wind_speed", "wind_direction"],
        full_idx=time_full_idx,
    )

    dfs["env"].station = dfs["env"].station.str.upper()
    dfs["env"] = dfs["env"].replace({"SHA TIN": "SHATIN"})

    dfs["idx"].station = dfs["idx"].station.str.upper()
    dfs["idx"] = dfs["idx"].replace({"SHA TIN": "SHATIN"})

    for key in ["env", "idx"]:
        dfs[key] = dfs[key].merge(stations_epd, on="station", how="inner")
    for key in ["hum", "tmp", "prs", "wnd"]:
        dfs[key] = dfs[key].merge(stations_hko[["station", "lat_idx", "lon_idx"]], on="station", how="inner")

    for df in dfs.values():
        df.drop(columns=["station", "Latitude", "Longitude", "station_code"], errors="ignore", inplace=True)

    from functools import reduce

    keys = ["env", "idx", "hum", "tmp", "prs", "wnd"]
    df_all48 = reduce(
        lambda left, right: pd.merge(left, right, on=["datetime", "lat_idx", "lon_idx"], how="outer"),
        [dfs[k] for k in keys],
    )

    month_to_season = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df_all48["season"] = df_all48["datetime"].dt.month.map(month_to_season)
    df_all48["is_weekend"] = df_all48["datetime"].dt.dayofweek.isin([5, 6]).astype(int)

    df_all48 = df_all48.loc[
        :,
        [
            "datetime",
            "so2",
            "no",
            "no2",
            "rsp",
            "o3",
            "fsp",
            "lat_idx",
            "lon_idx",
            "aqi",
            "humidity",
            "max_temp",
            "min_temp",
            "pressure",
            "wind_direction",
            "wind_speed",
            "max_wind_speed",
            "season",
            "is_weekend",
        ],
    ]
    lat_idxs = np.arange(46)
    lon_idxs = np.arange(68)

    ds48 = df_all48.set_index(["datetime", "lat_idx", "lon_idx"]).to_xarray()

    ds48 = ds48.reindex(datetime=time_full_idx, lat_idx=lat_idxs, lon_idx=lon_idxs)

    arr48 = ds48.to_array().transpose("datetime", "variable", "lat_idx", "lon_idx").values  # shape = (48, 16, H, W)

    def idw_fill(arr2d, mask, power=2, k=8, eps=1e-12):
        yy, xx = np.meshgrid(np.arange(arr2d.shape[0]), np.arange(arr2d.shape[1]), indexing="ij")
        pts = np.vstack((yy[~mask], xx[~mask])).T
        vals = arr2d[~mask]
        tree = cKDTree(pts)
        miss = np.vstack((yy[mask], xx[mask])).T
        dists, idxs = tree.query(miss, k=k, p=2)
        if k == 1:
            dists, idxs = dists[:, None], idxs[:, None]
        w = 1.0 / (dists**power + eps)
        w /= w.sum(axis=1, keepdims=True)
        filled = np.sum(w * vals[idxs], axis=1)
        out = arr2d.copy()
        out[mask] = filled
        return out

    def fill2d_griddata(arr2d, method="cubic"):
        yy, xx = np.meshgrid(np.arange(arr2d.shape[0]), np.arange(arr2d.shape[1]), indexing="ij")
        known = ~np.isnan(arr2d)
        pts = np.argwhere(known)
        vals = arr2d[known]
        out = griddata(pts, vals, (yy, xx), method=method)
        hole = np.isnan(out)
        if hole.any():
            out[hole] = idw_fill(out, hole)[hole]
        return out

    T, C, H, W = arr48.shape
    filled48 = np.empty_like(arr48)
    for t in range(T):
        for c in range(C):
            filled48[t, c] = fill2d_griddata(arr48[t, c])

    return filled48