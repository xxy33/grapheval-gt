import os
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report

from src.model_arch import GraphEvalGT,SimpleGCN 
from src.trainer import PaperGraphDataset, evaluate 
from src.config_loader import load_config, set_seed

def run_validation(config):
    set_seed(config['seed'])
    device = "cuda" if torch.cuda.is_available() and config['device'] == 'cuda' else "cpu"
    print(f"Validator using device: {device}")

    dataset_path = "data/processed/reviewadvisor_test"
    dataset = PaperGraphDataset(root=dataset_path)
    val_dataset = dataset
    
    if len(val_dataset) == 0:
        print("error")
        return

    model = SimpleGCN(config).to(device)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    base_save_path = config['training']['model_save_path']
    save_dir = os.path.dirname(base_save_path)
    base_filename = os.path.basename(base_save_path)
    dataset_name = os.path.basename(config['processed_data_paths']['iclr_papers'].rstrip('/'))
    model_path = os.path.join(save_dir, f"{dataset_name}_{base_filename}")
    
    if not os.path.exists(model_path):
        return    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(data.y.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    decision_map = config.get('decision_mapping', {})
    target_names = [name for name, idx in sorted(decision_map.items(), key=lambda item: item[1])]
    
    try:
        report = classification_report(all_labels, all_preds, zero_division=0, target_names=target_names)
    except ValueError:
        report = classification_report(all_labels, all_preds, zero_division=0)

    print(report)
    print("---------------------------------")
    
    return classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
