import torch
import torch.nn as nn

class MolecularLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, output_dim=1, n_layers=2, dropout=0.2):
        super(MolecularLSTM, self).__init__()
        
        # 1. Embedding Layer
        # Converts integer tokens (e.g., 15) to dense vectors (e.g., [0.1, -0.5, ...])
        # padding_idx=0 ensures the padding (0) is always ignored/zeroed out
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. LSTM Layer (The "Brain")
        # Bidirectional=True doubles the hidden_dim because it concatenates 
        # the Forward and Backward passes.
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # 3. Output Layer (Regression Head)
        # Input features = hidden_dim * 2 (because of bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Regularization to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len] (e.g., 32, 100)
        
        # A. Embed the integers
        embedded = self.embedding(x)  # Shape: [32, 100, 64]
        
        # B. Pass through LSTM
        # output contains the hidden states for every time step (atom)
        # hidden contains the final hidden state of the sequence
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # C. Feature Aggregation
        # We need one vector to represent the WHOLE molecule.
        # We take the final hidden state of the Forward pass and the Backward pass.
        # hidden[-2] = Last Forward State
        # hidden[-1] = Last Backward State
        combined_encoding = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # combined_encoding shape: [32, 256] (128*2)
        
        # D. Predict
        x = self.dropout(combined_encoding)
        prediction = self.fc(x) # Shape: [32, 1]
        
        return prediction

# --- Test Block ---
if __name__ == "__main__":
    # Quick sanity check
    vocab_size = 30 # Example size
    model = MolecularLSTM(vocab_size=vocab_size)
    
    # Fake input: Batch of 2 molecules, length 10
    dummy_input = torch.randint(0, vocab_size, (2, 10))
    
    output = model(dummy_input)
    print("\n--- Model Test ---")
    print(f"Dummy Input Shape: {dummy_input.shape}")
    print(f"Model Output Shape: {output.shape}") # Should be [2, 1]
    print("Model loaded successfully!")