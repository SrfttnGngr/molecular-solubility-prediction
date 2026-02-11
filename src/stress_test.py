import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw  # This contains MolsToGridImage
from captum.attr import LayerIntegratedGradients

from model import MolecularLSTM
from data_loader import SMILESTokenizer, get_data_loaders

# --- Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pth'

# ... (Keep get_prediction_and_attribution as is) ...
def get_prediction_and_attribution(model, tokenizer, smile):
    model.eval()
    model.zero_grad()
    
    # Encode
    indices = tokenizer.encode(smile)
    input_tensor = torch.tensor([indices], dtype=torch.long).to(DEVICE)
    
    # Predict Value
    with torch.no_grad():
        pred_value = model(input_tensor).item()
        
    # Explain (Attribution)
    lig = LayerIntegratedGradients(model, model.embedding)
    attributions, delta = lig.attribute(input_tensor, target=0, return_convergence_delta=True)
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    
    return pred_value, attributions.cpu().detach().numpy()

# --- FIXED FUNCTION ---
def visualize_comparison(results, test_name):
    """
    Draws a row of molecules to compare them side-by-side using MolsToGridImage.
    """
    mols = []
    legends = []
    all_highlights = []
    all_colors = []

    _, _, tokenizer = get_data_loaders()

    for smile, pred, attrs in results:
        mol = Chem.MolFromSmiles(smile)
        if not mol: continue
        mols.append(mol)
        legends.append(f"LogS: {pred:.2f}")
        
        # Map attributes to atoms
        highlights = []
        colors = {}
        tokens = [tokenizer.inv_vocab[idx] for idx in tokenizer.encode(smile) if idx != 0]
        atom_idx = 0
        
        for i, token in enumerate(tokens):
            if i >= len(attrs): break
            score = attrs[i]
            if token.isalpha() or '[' in token:
                if atom_idx < mol.GetNumAtoms():
                    # Threshold: Only highlight if score is significant (optional cleanup)
                    highlights.append(atom_idx)
                    
                    # Green = Soluble, Red = Insoluble
                    if score > 0:
                        colors[atom_idx] = (0.2, 1.0, 0.2) 
                    else:
                        colors[atom_idx] = (1.0, 0.2, 0.2)
                    atom_idx += 1
        
        all_highlights.append(highlights)
        all_colors.append(colors)

    # Draw Grid (Using the standard Python function)
    # returns a PIL image
    img = Draw.MolsToGridImage(
        mols, 
        molsPerRow=len(mols), 
        subImgSize=(300, 300),
        legends=legends,
        highlightAtomLists=all_highlights,
        highlightAtomColors=all_colors
    )
    
    filename = f"test_{test_name}.png"
    img.save(filename)
    print(f"Saved comparison: {filename}")

def main():
    # ... (Keep your existing main function exactly as it was) ...
    # Just ensure you paste the main() logic here again or keep it in the file.
    _, _, tokenizer = get_data_loaders()
    vocab_size = len(tokenizer.vocab) + 1
    model = MolecularLSTM(vocab_size, 64, 128)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.to(DEVICE)

    # --- TEST 1 ---
    print("\nRunning Test 1: Polarity Shift...")
    polarity_test = ["CCCCCC", "CCCCCO", "OCCCCO"]
    results = []
    for s in polarity_test:
        pred, attr = get_prediction_and_attribution(model, tokenizer, s)
        results.append((s, pred, attr))
    visualize_comparison(results, "Polarity_Shift")

    # --- TEST 2 ---
    print("Running Test 2: Halogen Effect...")
    halogen_test = ["c1ccccc1", "c1ccccc1Cl", "c1ccccc1(Cl)Cl"]
    results = []
    for s in halogen_test:
        pred, attr = get_prediction_and_attribution(model, tokenizer, s)
        results.append((s, pred, attr))
    visualize_comparison(results, "Halogen_Effect")

    # --- TEST 3 ---
    print("Running Test 3: Complex Molecule...")
    complex_test = ["CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"]
    results = []
    for s in complex_test:
        pred, attr = get_prediction_and_attribution(model, tokenizer, s)
        results.append((s, pred, attr))
    visualize_comparison(results, "Complex_Steroid")

if __name__ == "__main__":
    main()