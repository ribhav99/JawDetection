import torch
import torchaudio
import pandas as pd
from model import Wav2Vec2GRUModel

model_path = "../models/model_epoch_479.pt"
test_csv = "/Users/ribhavkapur/Desktop/clean_sirt/KYLE-D1-012.csv"
test_wav = "/Users/ribhavkapur/Desktop/clean_sirt/KYLE-D1-012.wav"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = Wav2Vec2GRUModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

df = pd.read_csv(test_csv)
waveform, sample_rate = torchaudio.load(test_wav)
waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
targets = torch.tensor(df["Value"].to_list())
mask = torch.ones(targets.size(0), dtype=torch.bool)
mask[::6] = False
targets_filtered = targets[mask]
with torch.no_grad():
    features = model(waveform, encode_input=True)

print(f"features_shape: {features.shape}")
print(f"targets_shape: {targets_filtered.shape}")

# Calculate the loss
criterion = torch.nn.MSELoss()
loss = criterion(features, targets_filtered)
print(f"Loss: {loss.item()}")
abs_loss = torch.abs(features - targets_filtered)
print(f"Mean absolute loss: {abs_loss.mean().item()}")
print(f"Max absolute loss: {abs_loss.max().item()}")
