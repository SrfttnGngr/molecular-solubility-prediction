import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from data_loader import get_data_loaders
from model import MolecularLSTM

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, iterator, optimizer, criterion):
    """
    The Training Loop: 
    1. Grabs a batch.
    2. Zeroes old gradients.
    3. Predicts.
    4. Calculates Loss.
    5. Backpropagates (learns).
    """
    model.train() # Set to training mode (enables Dropout)
    epoch_loss = 0
    
    for batch in iterator:
        # Move data to GPU if available
        src = batch['smiles_tensor'].to(DEVICE)
        trg = batch['target'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward Pass
        predictions = model(src).squeeze(1) # Remove extra dimension
        
        # Calculate Loss
        loss = criterion(predictions, trg)
        
        # Backward Pass (The Learning)
        loss.backward()
        
        # Update Weights
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    """
    The Validation Loop:
    Checks performance on unseen data. No learning happens here.
    """
    model.eval() # Set to evaluation mode (disables Dropout)
    epoch_loss = 0
    
    with torch.no_grad(): # Don't calculate gradients (saves memory)
        for batch in iterator:
            src = batch['smiles_tensor'].to(DEVICE)
            trg = batch['target'].to(DEVICE)

            predictions = model(src).squeeze(1)
            loss = criterion(predictions, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def main():
    # 1. Prepare Data
    print("Preparing Data...")
    train_loader, test_loader, tokenizer = get_data_loaders(BATCH_SIZE)
    vocab_size = len(tokenizer.vocab) + 1 # +1 for safety/padding index handling
    print(f"Vocab Size: {vocab_size}")

    # 2. Initialize Model
    model = MolecularLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    
    # 3. Setup Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Mean Squared Error for Regression

    # 4. Training Loop
    best_valid_loss = float('inf')
    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        valid_loss = evaluate(model, test_loader, criterion)
        
        # Save the model if it's the best one so far
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'\tEpoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} * (New Best)')
        else:
            print(f'\tEpoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    print(f"\nTraining Complete. Best Validation Loss: {best_valid_loss:.3f}")
    print("Model saved as 'best_model.pth'")

if __name__ == "__main__":
    main()