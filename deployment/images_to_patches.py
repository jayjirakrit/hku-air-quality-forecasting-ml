import numpy as np
import pandas as pd


def images_to_patches(images: str | np.ndarray, stations_filepath, patch_size=15):
    if isinstance(images, str):
        images = np.load(images)
    images = images.astype(np.float32)

    stations = pd.read_csv(stations_filepath).drop(columns=["Longitude", "Latitude"])
    station_cells = stations[["lat_idx", "lon_idx"]].to_numpy()

    pad = patch_size // 2
    images_padded = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="edge")

    windows = np.lib.stride_tricks.sliding_window_view(
        x=images_padded, window_shape=(patch_size, patch_size), axis=(2, 3)
    )
    i_idx = station_cells[:, 0]
    j_idx = station_cells[:, 1]

    all_patches = windows[:, :, i_idx, j_idx, :, :]

    return all_patches


if __name__ == "__main__":
    images_path = "./images_filled_griddata_idw_correct_date_aqi_weekend.npy"
    stations_filepath = "./data/stations_epd_idx.csv"
    all_patches = images_to_patches(images_path, stations_filepath)
    np.save("all_patches_15.npy", all_patches)
