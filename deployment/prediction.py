import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import json

from images_to_patches import images_to_patches
from model_architecture import AQI_CNNLSTM


# Thresholds for converting added health risk to AQHI bands
AQHI_THRESHOLDS = np.array([1.87, 3.73, 5.60, 7.46, 9.33, 11.20, 12.81, 14.94, 17.08, 19.21])


def load_images(path: str) -> np.ndarray:
    return np.load(path)


def load_scalers(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_station_names(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    return df["station"].tolist()


def transform_with_channel_scalers(X: np.ndarray, scalers) -> np.ndarray:
    T, C, S, H, W = X.shape
    Xs = np.zeros_like(X, dtype=np.float32)
    for c, sc in enumerate(scalers):
        vals = X[:, c].reshape(-1, 1)
        Xs[:, c] = sc.transform(vals).reshape(T, S, H, W)
    return Xs


def ar_to_aqhi(ar: np.ndarray) -> np.ndarray:
    # Convert %AR to AQHI index (1-10)
    bin_index = np.sum(ar[..., np.newaxis] > AQHI_THRESHOLDS, axis=-1) + 1
    return np.where(bin_index <= 10, bin_index, 10)


def prepare_input(images_path: str, scalers_path: str, stations_csv: str, patch_size: int) -> np.ndarray:
    images = load_images(images_path)
    patches = images_to_patches(images, stations_csv, patch_size)
    scalers = load_scalers(scalers_path)
    scaled = transform_with_channel_scalers(patches, scalers)
    # Reorder to (batch=stations, seq, channels, H, W)
    return scaled.transpose(2, 0, 1, 3, 4)


def load_model(path: str, device) -> nn.Module:
    model = AQI_CNNLSTM(
        in_channels=16,
        num_residual_units=4,
        lstm_hidden_size=128,
        num_lstm_layers=1,
        seq_length=48,
        pred_len=24,
    ).to(device)
    model.load_state_dict(torch.load(path,map_location=device))
    model.eval()
    return model


def predict(model: nn.Module, X_s: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        inp = torch.tensor(X_s).to(device)
        ar = model(inp).cpu().numpy()
    return ar


def format_output(aqhi: np.ndarray, station_names: list[str], start_hour: int = 1) -> list[dict]:
    hours = aqhi.shape[1]
    times = [f"{h:02d}:00" for h in range(start_hour, start_hour + hours)]
    output = []
    for si, station in enumerate(station_names[: aqhi.shape[0]]):
        for hi, t in enumerate(times):
            output.append({"date": None, "time": t, "station": station, "aqi": int(aqhi[si, hi]), "pm2_5": None})
    return output


def main():
    images_path = "./deployment/data/past48h_tensor.npy"
    scalers_path = "./deployment/data/x_scalers.pkl"
    stations_csv = "./deployment/data/stations_epd_idx.csv"
    patch_size = 15
    aqi_model_path = "./deployment/data/cnn_lstm_aqi.pth"
    fsp_mode_path = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_s = prepare_input(images_path, scalers_path, stations_csv, patch_size)
    aqi_model = load_model(aqi_model_path, device)
    ar = predict(aqi_model, X_s, device)
    aqhi = ar_to_aqhi(ar)
    station_names = load_station_names(stations_csv)
    output = format_output(aqhi, station_names)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
