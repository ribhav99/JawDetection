import torch
import torchaudio
import pandas as pd
from model import Wav2Vec2GRUModel
import os
from data_cleaning import audio_cuts
import matplotlib.pyplot as plt

model_path = os.path.join(os.path.dirname(__file__), "../models/model_epoch_736.pt")
file_name = "MichaelRosen_HotFood_clean"
test_csv = f'/Users/ribhavkapur/Desktop/everything/College/CS/JawDectection/audio_clips/{file_name}.csv'
test_wav = f'/Users/ribhavkapur/Desktop/everything/College/CS/JawDectection/audio_clips/{file_name}.wav'
ground_truth_multiplier = 10 # This is because the values for jaw from jali animation are between 0 and 1 whereas the values from SIRT are between 0 and 10

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
df["Value"] = df["Value"] * ground_truth_multiplier
waveform, sample_rate = torchaudio.load(test_wav)
waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
targets = torch.tensor(df["Value"].to_list())

# Slice targets and audio to align them perfectly
if file_name in audio_cuts:
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
if file_name in audio_cuts:
    starting = targets[:int(audio_cuts[file_name] * 60)]
else:
    starting = [] # HERE
middle = output
indices = torch.arange(len(middle))
duplicate_indices = (indices + 1) % 5 == 0
duplicated_values = middle[duplicate_indices]
middle = torch.cat((middle, duplicated_values))
middle = middle[torch.argsort(torch.cat((indices, indices[duplicate_indices] + 0.5)))]
new_output = torch.cat((starting, middle), dim=0)

new_output_path = os.path.join(os.path.dirname(__file__), f"../new_outputs/{file_name}.pt")
torch.save(new_output, new_output_path)

# Plot the output and targets with horizontal scroll
from matplotlib.widgets import Slider

fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Initial plot
line1, = ax.plot(output, label='Predicted')
line2, = ax.plot(targets_filtered, label='Targets')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Output vs Targets')
ax.legend()

# Slider for horizontal scrolling
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Scroll', 0, len(output) - 100, valinit=0, valstep=1)

def update(val):
    pos = slider.val
    ax.set_xlim(pos, pos + 100)
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
