import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
import wget
import os

# --- Configuration ---
DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
SAVE_PATH = "data/delaney-processed.csv"

class SMILESTokenizer:
    """
    Custom Tokenizer to handle chemical elements correctly (e.g., 'Cl' vs 'C', 'l').
    """
    def __init__(self):
        # Regex to capture elements (Br, Cl, Si, etc.), brackets, numbers, and symbols
        # This prevents splitting 'Cl' into 'C' and 'l'
        self.token_pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.vocab = {}
        self.inv_vocab = {}
    
    def build_vocab(self, smiles_list):
        """Creates a mapping of {token: integer}."""
        unique_tokens = set()
        for smile in smiles_list:
            tokens = re.findall(self.token_pattern, smile)
            unique_tokens.update(tokens)
        
        # specific tokens
        self.vocab = {token: i+1 for i, token in enumerate(sorted(unique_tokens))}
        self.vocab["<PAD>"] = 0  # Padding token
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        print(f"Vocabulary Size: {len(self.vocab)}")
        
    def encode(self, smile, max_len=100):
        """Converts a SMILES string into a list of integers with padding."""
        tokens = re.findall(self.token_pattern, smile)
        encoded = [self.vocab[t] for t in tokens if t in self.vocab]
        
        # Padding or Truncating
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded)) # Pad with 0
        else:
            encoded = encoded[:max_len] # Truncate
            
        return encoded

class MoleculeDataset(Dataset):
    """
    PyTorch Dataset wrapper for the ESOL dataset.
    """
    def __init__(self, csv_file, tokenizer, max_len=100):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # The column names in ESOL dataset are usually:
        # 'smiles' (input) and 'measured log solubility in mols per litre' (target)
        self.smiles = self.data['smiles'].values
        self.targets = self.data['measured log solubility in mols per litre'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        target = self.targets[idx]
        
        # Encode SMILES to tensor
        encoded_smile = self.tokenizer.encode(smile, self.max_len)
        
        return {
            "smiles_tensor": torch.tensor(encoded_smile, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
            "raw_smiles": smile  # Useful for debugging/visualization later
        }

def get_data_loaders(batch_size=32, test_split=0.2):
    # 1. Download Data if not exists
    if not os.path.exists(SAVE_PATH):
        os.makedirs("data", exist_ok=True)
        print("Downloading ESOL dataset...")
        wget.download(DATA_URL, SAVE_PATH)
        print("\nDownload complete.")

    # 2. Load Raw Data to build vocab
    df = pd.read_csv(SAVE_PATH)
    tokenizer = SMILESTokenizer()
    tokenizer.build_vocab(df['smiles'].values)

    # 3. Create Dataset
    dataset = MoleculeDataset(SAVE_PATH, tokenizer)

    # 4. Split into Train/Test
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 5. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer

# --- Test Block (Run this file directly to check) ---
if __name__ == "__main__":
    tr_loader, te_loader, tok = get_data_loaders()
    
    # Grab a sample batch
    sample = next(iter(tr_loader))
    print("\n--- Sample Batch ---")
    print(f"Shape of Input (Batch, Seq_Len): {sample['smiles_tensor'].shape}")
    print(f"Shape of Target (Batch): {sample['target'].shape}")
    print(f"Sample Raw SMILES: {sample['raw_smiles'][0]}")
    print(f"Sample Encoded Tensor: {sample['smiles_tensor'][0]}")