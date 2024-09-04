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
def load_singular_data(folder, file_name, input_length=2000):
    '''
    # csv and audio file must be in the same folder
    # the filenames must be the same, one ending in .wav, the other in .csv
    # input_length is the desired duration in milliseconds
    '''
    df = pd.read_csv(os.path.join(folder, file_name + '.csv'))
    waveform, sample_rate = torchaudio.load(os.path.join(folder, file_name + '.wav'))
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    targets = torch.tensor(df["Value"].to_list())
    mask = torch.ones(targets.size(0), dtype=torch.bool)
    mask[::6] = False 
    targets_filtered = targets[mask]
    with torch.no_grad():
        features = encoder(waveform).last_hidden_state.squeeze()  # Shape: [sequence_length, hidden_size]
    if targets_filtered.shape[0] > features.shape[0]:
        targets_filtered = targets_filtered[:features.shape[0]]
    elif targets_filtered.shape[0] < features.shape[0]:
        features = features[:targets_filtered.shape[0]]

    # each row in features corresponds to ~20ms of audio
    # we want to convert features into shape [num_sequences, sequence_length, hidden_size]
    # where sequence_length duration corresponds to input_length duration
    # if input length = 2000, then sequence_length = 2000/1000 * 50 = 100
    # so shape of new features = [features.shape[0]//100 + 1, 100, features.shape[1]]
    # Calculate the sequence length based on the input_length (2000ms corresponds to 100 timesteps)

    sequence_length = (input_length // 1000) * 50  # 50 frames per second
    num_sequences = features.shape[0] // sequence_length + 1
    padding_length = sequence_length * num_sequences - features.shape[0]
    features = torch.nn.functional.pad(features, (0, 0, 0, padding_length))
    features = features.reshape(num_sequences, sequence_length, features.shape[1])
    targets_filtered = torch.nn.functional.pad(targets_filtered, (0, padding_length))
    targets_filtered = targets_filtered.reshape(num_sequences, sequence_length)
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
    # Windows: C:/Users/Ribhav/Downloads/clean_sirt
    # Mac: /Users/ribhavkapur/Desktop/clean_sirt
    dataloader = prepare_dataloader('/Users/ribhavkapur/Desktop/clean_sirt', batch_size=32)
    for inputs, targets in dataloader:
        output = model(inputs, encode_input=False)
        print(f"output_shape: {output.shape}")
        print(f"input_shape: {inputs.shape}")
        print(f"target_shape: {targets.shape}")

        break
    # features, targets = load_singular_data('/Users/ribhavkapur/Desktop/clean_sirt', 'KYLE-D1-003')
    # print(features.shape)
    # print(targets.shape)