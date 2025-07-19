import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import json

from images_to_patches import images_to_patches
from model_architecture import AQI_CNNLSTM, FSP_CNNLSTM


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


def transform_with_scalers(X: np.ndarray, scaler) -> np.ndarray:
    flat = X.transpose(0, 2, 1, 3, 4).reshape(-1, 15 * 15 * 15)
    scaled = scaler.transform(flat)
    return scaled.reshape(X.shape[0], X.shape[2], 15, 15, 15).transpose(0, 2, 1, 3, 4)


def ar_to_aqhi(ar: np.ndarray) -> np.ndarray:
    # Convert %AR to AQHI index (1-10)
    bin_index = np.sum(ar[..., np.newaxis] > AQHI_THRESHOLDS, axis=-1) + 1
    return np.where(bin_index <= 10, bin_index, 10)


def prepare_input_aqi(images: np.ndarray, scalers_path: str, stations_csv: str, patch_size: int) -> np.ndarray:
    patches = images_to_patches(images, stations_csv, patch_size)
    scalers = load_scalers(scalers_path)
    scaled = transform_with_channel_scalers(patches, scalers)
    # Reorder to (batch=stations, seq, channels, H, W)
    return scaled.transpose(2, 0, 1, 3, 4)


def prepare_input_fsp(images: np.ndarray, scalers_path: str, stations_csv: str, patch_size: int) -> np.ndarray:
    # Remove the AQI channel
    images = np.delete(images, 6, axis=1)
    patches = images_to_patches(images, stations_csv, patch_size)
    scalers = load_scalers(scalers_path)
    scaled = transform_with_scalers(patches, scalers)
    # Reorder to (batch=stations, seq, channels, H, W)
    return scaled.transpose(2, 0, 1, 3, 4)


def load_model(aqi_path: str, pm_path, device) -> nn.Module:
    aqi_model = AQI_CNNLSTM(
        in_channels=16,
        num_residual_units=4,
        lstm_hidden_size=128,
        num_lstm_layers=1,
        seq_length=48,
        pred_len=24,
    ).to(device)
    fsp_model = FSP_CNNLSTM(
        n_stations=17,
        in_channels=15,
        cnn_embed=256,
        lstm_hidden=64,
        pred_len=24,
        embed_dim=16,
    ).to(device)
    aqi_model.load_state_dict(torch.load(aqi_path, map_location=device))
    aqi_model.eval()
    fsp_model.load_state_dict(torch.load(pm_path, map_location=device))
    fsp_model.eval()
    return aqi_model, fsp_model


def predict_aqi(model: nn.Module, X_s: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        inp = torch.tensor(X_s).to(device)
        ar = model(inp).cpu().numpy()
    return ar


def predict_fsp(model: nn.Module, X_s: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        inp = torch.tensor(X_s).to(device)
        station_idx = torch.tensor(list(range(17))).to(device)
        ar = model(inp, station_idx).cpu().numpy()
    return ar


def format_output(aqhi: np.ndarray, fsp: np.ndarray, station_names: list[str], start_hour: int = 1) -> list[dict]:
    hours = aqhi.shape[1]
    times = [f"{h:02d}:00" for h in range(start_hour, start_hour + hours)]
    output = []
    for si, station in enumerate(station_names[: aqhi.shape[0]]):
        for hi, t in enumerate(times):
            output.append(
                {"date": None, "time": t, "station": station, "aqi": int(aqhi[si, hi]), "pm2_5": (float(fsp[si, hi]))}
            )
    return output


def main():
    images_path = "./deployment/data/past48h_tensor.npy"
    aqi_scalers_path = "./deployment/data/x_scalers_aqi.pkl"
    fsp_scalers_path = "./deployment/data/x_scalers_fsp.pkl"
    stations_csv = "./deployment/data/stations_epd_idx.csv"
    aqi_model_path = "./deployment/data/cnn_lstm_aqi.pth"
    fsp_model_path = "./deployment/data/cnn_lstm_fsp.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = load_images(images_path)
    X_s_aqi = prepare_input_aqi(images, aqi_scalers_path, stations_csv, 3)
    X_s_fsp = prepare_input_fsp(images, fsp_scalers_path, stations_csv, 15)
    aqi_model, fsp_model = load_model(aqi_model_path, fsp_model_path, device)
    ar = predict_aqi(aqi_model, X_s_aqi, device)
    fsp = predict_fsp(fsp_model, X_s_fsp, device)
    aqhi = ar_to_aqhi(ar)
    station_names = load_station_names(stations_csv)
    output = format_output(aqhi, fsp, station_names)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
