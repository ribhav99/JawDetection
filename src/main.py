from model import Wav2Vec2GRUModel
from data_cleaning import prepare_dataloader
from train import train_model
import torch

torch.manual_seed(6)

model = Wav2Vec2GRUModel()
# Windows: C:/Users/Ribhav/Downloads/clean_sirt
# Mac: /Users/ribhavkapur/Desktop/clean_sirt
dataloader = prepare_dataloader("C:/Users/Ribhav/Downloads/clean_sirt", batch_size=512, parallel_processing=False, skip=[6,12,14,3,4,5,7,8,9,10,11,13])
criterion = torch.nn.MSELoss()
train_model(model, dataloader, criterion, num_epochs=1000, use_wandb=True, patience=25)
