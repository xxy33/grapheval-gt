# src/trainer.py

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any
import numpy as np
from src.model_arch import GraphEvalGT,SimpleGCN 
from src.config_loader import load_config, set_seed

class PaperGraphDataset(Dataset):
    """Custom PyG Dataset to load preprocessed graph files."""
    def __init__(self, root, transform=None, pre_transform=None):
        super(PaperGraphDataset, self).__init__(root, transform, pre_transform)
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return sorted([f for f in os.listdir(self.root) if f.endswith('.pt')])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.root, self.processed_file_names[idx]),weights_only=False)
        return data

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        binary_labels = data.y.clone()
        binary_labels[binary_labels > 0] = 1  
        accept_logits,_ = torch.max(out[:,1:],dim=1)
        binary_out = torch.stack([out[:,0],accept_logits],dim=1)
        loss = F.cross_entropy(binary_out, binary_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        accept_logits,_ = torch.max(out[:,1:],dim=1)
        binary_out = torch.stack([out[:,0],accept_logits],dim=1)

        preds = binary_out.argmax(dim=1)
        binary_labels = data.y.clone()
        binary_labels[binary_labels > 0] = 1        
        all_preds.append(preds.cpu())
        all_labels.append(binary_labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return acc, macro_f1

def run_training(config: Dict[str, Any]):
    """Main function to run the training and validation process."""
    set_seed(config['seed'])
    device = "cuda" if torch.cuda.is_available() and config['device'] == 'cuda' else "cpu"
    print(f"Trainer using device: {device}")

    dataset_path = config['processed_data_paths']['iclr_papers'] 
    dataset = PaperGraphDataset(root=dataset_path)
    
    torch.manual_seed(config['seed']) 
    dataset = dataset.shuffle()
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size :]
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    
    model = SimpleGCN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)
    
    best_val_f1 = 0.0
    dataset_name = os.path.basename(config['processed_data_paths']['iclr_papers'].rstrip('/'))
    for epoch in range(1, config['training']['epochs'] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step(val_f1)
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            base_save_path = config['training']['model_save_path']
            save_dir = os.path.dirname(base_save_path)
            base_filename = os.path.basename(base_save_path)
            new_filename = f"{dataset_name}_{base_filename}"
            model_save_full_path = os.path.join(save_dir,new_filename)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), model_save_full_path)
            print(f"Saved new best model with F1: {best_val_f1:.4f}")

    print("Training finished.")
