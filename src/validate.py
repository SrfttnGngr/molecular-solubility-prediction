import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from captum.attr import LayerIntegratedGradients

from model import MolecularLSTM
from data_loader import SMILESTokenizer, get_data_loaders

# --- Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pth'
GRID_ROWS = 4
GRID_COLS = 3
GRID_SIZE = GRID_ROWS * GRID_COLS 

def get_attribution(model, tokenizer, smile):
    model.eval()
    model.zero_grad()
    indices = tokenizer.encode(smile)
    input_tensor = torch.tensor([indices], dtype=torch.long).to(DEVICE)
    lig = LayerIntegratedGradients(model, model.embedding)
    attributions, delta = lig.attribute(input_tensor, target=0, return_convergence_delta=True)
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions.cpu().detach().numpy()

def draw_single_molecule(mol, highlights, colors):
    # Setup Canvas
    width, height = 500, 500
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    
    # Custom Drawing Options
    opts = drawer.drawOptions()
    opts.bondLineWidth = 3
    opts.highlightBondWidthMultiplier = 1
    opts.highlightRadius = 0.4
    opts.clearBackground = False 
    drawer.SetFontSize(1.0) 

    # Draw
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, 
        mol, 
        highlightAtoms=highlights, 
        highlightAtomColors=colors
    )
    drawer.FinishDrawing()
    
    # Convert to Image
    png_data = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png_data))

def main():
    print("Loading Test Data...")
    _, test_loader, tokenizer = get_data_loaders(batch_size=1) 
    vocab_size = len(tokenizer.vocab) + 1
    
    model = MolecularLSTM(vocab_size, 64, 128).to(DEVICE)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    print(f"Calculating Metrics...")
    all_preds = []
    all_reals = []
    viz_candidates = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            src = batch['smiles_tensor'].to(DEVICE)
            trg = batch['target'].item()
            raw_smile = batch['raw_smiles'][0]
            pred = model(src).item()
            all_preds.append(pred)
            all_reals.append(trg)
            
            if len(viz_candidates) < GRID_SIZE:
                viz_candidates.append((raw_smile, trg, pred))

    r2 = r2_score(all_reals, all_preds)
    mae = mean_absolute_error(all_reals, all_preds)
    rmse = np.sqrt(mean_squared_error(all_reals, all_preds))

    print("\n" + "="*40)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"MAE:      {mae:.4f}")
    print("="*40 + "\n")

    print(f"Generating clean grid...")
    # Increase Figure Height to allow room for text at bottom
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(15, 22))
    axes = axes.flatten()
    
    for i, (smile, real, pred) in enumerate(viz_candidates):
        ax = axes[i]
        mol = Chem.MolFromSmiles(smile)
        
        if not mol:
            ax.axis('off')
            continue

        # Color Logic
        attrs = get_attribution(model, tokenizer, smile)
        highlights = []
        colors = {}
        tokens = [tokenizer.inv_vocab[idx] for idx in tokenizer.encode(smile) if idx != 0]
        atom_idx = 0
        
        for k, token in enumerate(tokens):
            if k >= len(attrs): break
            score = attrs[k]
            if token.isalpha() or '[' in token:
                if atom_idx < mol.GetNumAtoms():
                    if abs(score) > 0.05: 
                        highlights.append(atom_idx)
                        if score > 0:
                            colors[atom_idx] = (0.6, 1.0, 0.6) # Pastel Green
                        else:
                            colors[atom_idx] = (1.0, 0.6, 0.6) # Pastel Red
                    atom_idx += 1
        
        img = draw_single_molecule(mol, highlights, colors)
        
        ax.imshow(img)
        ax.axis('off')
        
        # --- THE FIX: Text Below Image ---
        diff = abs(real - pred)
        label_text = f"Real: {real:.2f} | Pred: {pred:.2f}\nDiff: {diff:.2f}"
        
        text_color = 'black'
        if diff > 1.0: text_color = 'red'
        
        # Place text at coordinate (0.5, -0.1) relative to the subplot
        # 0.5 = Center Horizontal, -0.1 = Below the bottom axis
        ax.text(0.5, -0.1, label_text, 
                fontsize=14, 
                fontfamily='sans-serif', 
                fontweight='bold', 
                color=text_color, 
                ha='center', 
                va='top', 
                transform=ax.transAxes)

    # Adjust layout to prevent overlap since we added text outside the axes
    plt.tight_layout()
    # Add a bit more spacing between rows to accommodate the bottom text
    plt.subplots_adjust(hspace=0.3) 
    
    plt.savefig("large_grid.png", dpi=150)
    print(f"Saved: large_grid.png")

if __name__ == "__main__":
    main()