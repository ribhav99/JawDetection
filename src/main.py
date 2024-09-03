from model import Wav2Vec2GRUModel
from data_cleaning import prepare_dataloader
from train import train_model
import torch

torch.manual_seed(6)

model = Wav2Vec2GRUModel()
dataloader = prepare_dataloader("/Users/ribhavkapur/Desktop/clean_sirt", batch_size=512, parallel_processing=False)
criterion = torch.nn.MSELoss()
train_model(model, dataloader, criterion, num_epochs=100, use_wandb=True, patience=15)
