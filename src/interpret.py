import torch
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D  # Fixed Import
from captum.attr import LayerIntegratedGradients

from model import MolecularLSTM
from data_loader import SMILESTokenizer, get_data_loaders

# --- Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pth'

def get_attribution(model, tokenizer, smile):
    """
    Uses Integrated Gradients to find which atoms affect the prediction.
    """
    model.eval()
    model.zero_grad()

    # 1. Prepare Input
    input_indices = tokenizer.encode(smile)
    # We need a batch dimension (1, seq_len)
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(DEVICE)
    
    # 2. Setup Captum
    # We target the 'embedding' layer
    lig = LayerIntegratedGradients(model, model.embedding)

    # 3. Calculate Attributions
    # target=0 implies the first output dimension (we only have 1)
    attributions, delta = lig.attribute(input_tensor, target=0, return_convergence_delta=True)
    
    # Attributions shape: [1, seq_len, embedding_dim]
    # Sum across embedding dimension to get one score per token
    attributions = attributions.sum(dim=2).squeeze(0)
    
    # Normalize for visualization
    attributions = attributions / torch.norm(attributions)
    
    return attributions.cpu().detach().numpy(), input_indices

def visualize_molecule(smile, attributions, tokenizer, save_name="molecule_heatmap.png"):
    """
    Draws the molecule and highlights atoms based on attribution scores.
    """
    mol = Chem.MolFromSmiles(smile)
    if not mol:
        print(f"Invalid SMILES: {smile}")
        return

    # Prepare highlighting containers
    highlight_atoms = []
    highlight_colors = {}
    
    # Get raw tokens (ignoring padding)
    tokens = [tokenizer.inv_vocab[idx] for idx in tokenizer.encode(smile) if idx != 0]
    
    # RDKit Atom Iterator (to map tokens to atoms)
    atom_idx = 0
    
    # Iterate through tokens
    for i, token in enumerate(tokens):
        if i >= len(attributions): break
        
        score = attributions[i]
        
        # If the token is likely an atom (e.g., 'C', 'O', 'Cl', '[NH]')
        # We assume it maps to the next atom in the RDKit molecule object
        if token.isalpha() or '[' in token: 
            if atom_idx < mol.GetNumAtoms():
                highlight_atoms.append(atom_idx)
                
                # Color Logic:
                # Green = Increases Solubility (Positive contribution)
                # Red   = Decreases Solubility (Negative contribution)
                if score > 0:
                    base_color = (0.2, 1.0, 0.2) # Light Green
                else:
                    base_color = (1.0, 0.2, 0.2) # Light Red
                
                # Dictionary expects {atom_index: (r,g,b)}
                highlight_colors[atom_idx] = base_color
                
                atom_idx += 1

    # --- DRAWING (FIXED SECTION) ---
    try:
        # Use Cairo for PNG support
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
        
        # Prepare the molecule for drawing
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, 
            mol, 
            highlightAtoms=highlight_atoms, 
            highlightAtomColors=highlight_colors
        )
        
        drawer.FinishDrawing()
        
        # Save to file
        with open(save_name, 'wb') as f:
            f.write(drawer.GetDrawingText())
        
        print(f"Saved visualization to {save_name}")
        
    except Exception as e:
        print(f"Error drawing molecule: {e}")

def main():
    # 1. Load Data & Vocab to get correct vocabulary size
    _, _, tokenizer = get_data_loaders()
    vocab_size = len(tokenizer.vocab) + 1
    
    # 2. Load Model
    # Important: Use same dimensions as training!
    model = MolecularLSTM(vocab_size, embedding_dim=64, hidden_dim=128)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    else:
        print("Error: best_model.pth not found. Please run train.py first.")
        return

    model.to(DEVICE)
    
    # 3. Test Molecules
    # Ethanol (Soluble), Heptane (Insoluble), Benzoic Acid
    test_smiles = [
        "CCO",          
        "CCCCCCC",      
        "c1ccccc1C(=O)O" 
    ]
    
    print("\n--- Generating Interpretability Maps ---")
    for i, smile in enumerate(test_smiles):
        # Predict
        indices = torch.tensor([tokenizer.encode(smile)], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            pred = model(indices).item()
            
        print(f"\nMolecule: {smile}")
        print(f"Predicted Log Solubility: {pred:.3f}")
        
        # Explain
        attrs, _ = get_attribution(model, tokenizer, smile)
        
        # Sanitize filename
        safe_name = smile.replace("(", "").replace(")", "").replace("=", "")
        visualize_molecule(smile, attrs, tokenizer, save_name=f"viz_{i}_{safe_name}.png")

if __name__ == "__main__":
    main()