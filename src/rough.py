import torch
import torchaudio
import pandas as pd
import os

# DON'T USE TAKE 6
folder = "/Users/ribhavkapur/Desktop/clean_sirt"
filename = "KYLE-D1-001"

# waveform, sample_rate = torchaudio.load(f"{folder}/{filename}.wav")
# waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
# waveform = waveform.squeeze()
# print(f"Original waveform shape: {waveform.shape}")
# print(f"Original waveform shape after encoding: {waveform.shape[0] // 320}")
# waveform = waveform[16000 * 7:]
# print(f"Waveform shape: {waveform.shape}")
# print(f"Waveform shape after encoding: {waveform.shape[0] // 320}")

# df = pd.read_csv(f"{folder}/{filename}.csv")
# targets = torch.tensor(df["Value"].to_list())
# mask = torch.ones(targets.size(0), dtype=torch.bool)
# mask[::6] = False
# targets_filtered = targets[mask]
# print(f"Ones in mask: {mask.sum()}")
# print(f"Zeros in mask: {(~mask).sum()}")
# print(f"Targets shape before mask: {targets.shape}")
# print(f"Targets shape after mask: {targets_filtered.shape}")

d = {'KYLE-D1-001': 421/60, 'KYLE-D1-002': 380/60, 'KYLE-D1-003': 310/60, 'KYLE-D1-004': 457/60, \
     'KYLE-D1-005': 419/60, 'KYLE-D1-006': 388/60, 'KYLE-D1-007': 510/60, 'KYLE-D1-008': 382/60, \
     'KYLE-D1-009': 356/60, 'KYLE-D1-010': 320/60, 'KYLE-D1-011': 419/60, 'KYLE-D1-012': 363/60, \
     'KYLE-D1-013': 176/60, 'KYLE-D1-014': 284/60}

for f in os.listdir(folder):
    # if f.endswith('.wav'):
        # filename = f.split(".")[0]
        # waveform, sample_rate = torchaudio.load(f"{folder}/{filename}.wav")
        # waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        # waveform = waveform.squeeze()
        # waveform = waveform[int(d[filename] * 16000):]
        # print(f"Waveform shape for {filename}: {waveform.shape}")
        # df = pd.read_csv(f"{folder}/{filename}.csv")
        # targets = torch.tensor(df["Value"].to_list())
        # # targets = targets[int(d[filename]*60):]
        # mask = torch.ones(targets.size(0), dtype=torch.bool)
        # mask[::6] = False
        # targets = targets[mask]
        # print(f"Targets shape for {filename}: {targets.shape}")
        # print(f"Difference in lengths: {waveform.shape[0]//320 - targets.shape[0]}")
        # print("\n\n")
    if f.endswith('.csv'):
        # predict audio legnth based on num csv rows
        filename = f.split(".")[0]
        df = pd.read_csv(f"{folder}/{filename}.csv")
        print(f"Predicted audio length for {filename}: {df.shape[0] / 60}s")
        waveform, sample_rate = torchaudio.load(f"{folder}/{filename}.wav")
        print(f"Actual audio length for {filename}: {waveform.shape[1] / sample_rate}s")
        print(f"Audio length after cut: {(waveform.shape[1] / sample_rate) - d[filename]}s")
        print("\n")

