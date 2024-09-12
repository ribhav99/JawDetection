import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2GRUModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=3, output_size=1):
        super(Wav2Vec2GRUModel, self).__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        self.gru = nn.GRU(input_size=self.wav2vec2.config.hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=True,
                          batch_first=True)
        # self.pooling = nn.AvgPool1d(kernel_size=2, stride=5)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Bidirectional GRU doubles hidden size
        self.relu = nn.ReLU()
        
    def forward(self, input_values, encode_input=False):
        if encode_input:
            with torch.no_grad():  # Freeze the Wav2Vec2 model
                features = self.wav2vec2(input_values).last_hidden_state.squeeze()  # Shape: [batch_size, sequence_length, hidden_size]
        else:
            features = input_values
        gru_output, _ = self.gru(features)  # Shape: [batch_size, sequence_length, hidden_size * 2]
        
        # pooled_output = self.pooling(gru_output.permute(0, 2, 1))  # Permute to (batch_size, hidden_size * 2, sequence_length)
        # pooled_output = pooled_output.permute(0, 2, 1)  # Permute back to (batch_size, sequence_length, hidden_size * 2)
        
        controller_activation = self.fc(gru_output)  # Shape: [batch_size, new_sequence_length, output_size]
        controller_activation = self.relu(controller_activation)
        
        return controller_activation.squeeze(-1)  # Shape: [batch_size, sequence_length]

if __name__ == '__main__':
    import torchaudio
    audio_file = "/Users/ribhavkapur/Desktop/clean_sirt/KYLE-D1-001.wav"
    model = Wav2Vec2GRUModel()
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    print(f"audio shape: {waveform.shape}")
    output = model(waveform, encode_input=True)

    print(f"output_shape: {output.shape}")  # Output will be a tensor with shape [8, 1], predicting controller activation for each sample


# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # Example target output
# target = torch.tensor([[5.0], [7.2], [3.8], [9.1], [2.5], [6.3], [4.4], [8.9]])

# # Forward pass
# output = model(input_values)

# # Compute loss
# loss = criterion(output, target)
