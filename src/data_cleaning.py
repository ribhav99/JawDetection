import pandas as pd
import os
import torchaudio
from transformers import Wav2Vec2Model
import torch
from tqdm import tqdm
from pqdm.processes import pqdm
import os
import logging
import warnings
from torch.utils.data import TensorDataset, DataLoader

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore", 
    message="Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 Transformers."
)

encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#TODO: account for the lag/offset in the audio and mocap data
# it might be best to clean the data at the source instead of doing it in code here
def load_singular_data(folder, file_name):
    '''
    # csv and audio file must be in the same folder
    # the filenames must be the same, one ending in .wav, the other in .csv
    '''
    df = pd.read_csv(os.path.join(folder, file_name + '.csv'))
    waveform, sample_rate = torchaudio.load(os.path.join(folder, file_name + '.wav'))
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    targets = torch.tensor(df["Value"].to_list())
    mask = torch.ones(targets.size(0), dtype=torch.bool)
    mask[::6] = False 
    targets_filtered = targets[mask]
    with torch.no_grad():
        features = encoder(waveform).last_hidden_state.squeeze()  # Shape: [batch_size, sequence_length, hidden_size]
    if targets_filtered.shape[0] > features.shape[0]:
        targets_filtered = targets_filtered[:features.shape[0]]
    elif targets_filtered.shape[0] < features.shape[0]:
        features = features[:targets_filtered.shape[0]]

    # print(f"audio_shape: {waveform.shape}")
    # print(f"target_shape: {targets_filtered.shape}")
    # print(f"encoded_shape: {features.shape}")

    return features, targets_filtered 


def prepare_dataloader(folder, batch_size=32, parallel_processing=False, skip=None):
    # skip is a list of ints or None
    filenames = [f.split(".")[0] for f in os.listdir(folder) if f.endswith('.wav')]
    input_data = [(folder, filename) for filename in filenames]
    if skip:
        for j in skip:
            input_data = [i for i in input_data if str(j) not in i[1]]
    if not parallel_processing:
        results = [load_singular_data(*data) for data in tqdm(input_data)]
    else:
        results = pqdm(input_data, load_singular_data, n_jobs=os.cpu_count(), argument_type='args')
    features = torch.cat([r[0] for r in results], dim=0)
    targets = torch.cat([r[1] for r in results], dim=0)

    dataset = TensorDataset(features, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

if __name__ == "__main__":
    from model import Wav2Vec2GRUModel
    model = Wav2Vec2GRUModel()
    dataloader = prepare_dataloader('/Users/ribhavkapur/Desktop/clean_sirt', batch_size=32)
    for inputs, targets in dataloader:
        output = model(inputs, encode_input=False)
        print(output)
        print(f"output_shape: {output.shape}")
        break