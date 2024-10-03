import torch
import torchaudio
import pandas as pd
from model import Wav2Vec2GRUModel
import os
from data_cleaning import audio_cuts
import matplotlib.pyplot as plt

model_path = os.path.join(os.path.dirname(__file__), "../models/model_epoch_736.pt")
file_name = "KYLE-D1-012"
test_csv = f"/Users/ribhavkapur/Desktop/clean_sirt/{file_name}.csv"
test_wav = f"/Users/ribhavkapur/Desktop/clean_sirt/{file_name}.wav"
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

# Slice targets and audio to align them perfectly
waveform = waveform[:, int(audio_cuts[file_name] * 16000):]
targets = targets[int(audio_cuts[file_name] * 60):]
expected_length_from_audio = waveform.shape[1] / 16000
expected_length_from_targets = targets.shape[0] / 60
if expected_length_from_audio > expected_length_from_targets:
    waveform = waveform[:, :int(expected_length_from_targets * 16000)]
elif expected_length_from_audio < expected_length_from_targets:
    targets = targets[:int(expected_length_from_audio * 60)]

mask = torch.ones(targets.size(0), dtype=torch.bool)
mask[::6] = False
targets_filtered = targets[mask]
with torch.no_grad():
    output = model(waveform, encode_input=True)

print(f"output_shape: {output.shape}")
print(f"targets_shape: {targets_filtered.shape}")
if targets_filtered.shape[0] > output.shape[0]:
    targets_filtered = targets_filtered[:output.shape[0]]
elif targets_filtered.shape[0] < output.shape[0]:
    output = output[:targets_filtered.shape[0]]


# Calculate the loss
criterion = torch.nn.MSELoss()
loss = criterion(output, targets_filtered)
print(f"Loss: {loss.item()}")
abs_loss = torch.abs(output - targets_filtered)
print(f"Mean absolute loss: {abs_loss.mean().item()}")
print(f"Max absolute loss: {abs_loss.max().item()}")

# Create new output by substituting the targets with the predicted values
starting = targets[:int(audio_cuts[file_name] * 60)]
middle = output
indices = torch.arange(len(middle))
duplicate_indices = (indices + 1) % 5 == 0
duplicated_values = middle[duplicate_indices]
middle = torch.cat((middle, duplicated_values))
middle = middle[torch.argsort(torch.cat((indices, indices[duplicate_indices] + 0.5)))]
new_output = torch.cat((starting, middle), dim=0)

new_output_path = os.path.join(os.path.dirname(__file__), f"../new_outputs/{file_name}.pt")
torch.save(new_output, new_output_path)

# Plot the output and targets
plt.figure(figsize=(12, 6))
plt.plot(output, label='Predicted')
plt.plot(targets_filtered, label='Targets')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Output vs Targets')
plt.legend()
plt.show()
