import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import trange

def train_model(model, data_loader, criterion, num_epochs=100, use_wandb=False, patience=15):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(f"{current_dir}/../models"):
        os.makedirs(f"{current_dir}/../models")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    
    if use_wandb:
        wandb.init(project="jaw_model")
        wandb.watch(model, criterion, log="all", log_freq=10)

    prev_model_name = None
    for epoch in trange(num_epochs):
        if early_stop:
            print("Early stopping triggered")
            break
            
        model.train()
        running_loss = 0.0
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs, encode_input=False)
            assert outputs.shape == targets.shape
           
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        running_loss /= len(data_loader)
        
        if use_wandb:
            wandb.log({"epoch": epoch + 1, "loss": running_loss})
        
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")
        
        # Early stopping check
        if running_loss < best_loss:
            best_loss = running_loss
            epochs_no_improve = 0
            if prev_model_name is not None:
                os.remove(prev_model_name)
            prev_model_name = os.path.join(current_dir, "..", "models", f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), os.path.abspath(prev_model_name))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
    if use_wandb:
        wandb.save(prev_model_name, policy="now")
        # wandb.finish()
