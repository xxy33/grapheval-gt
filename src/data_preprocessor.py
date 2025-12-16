# src/data_preprocessor.py

import os
import json
import torch
import networkx as nx
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

# Import from our own project modules
from config_loader import load_config, set_seed

FILE_LIST = [
    "ai_researcher_test","ai_researcher_train","iclr_test","iclr_train","reviewadvisor_test",
    "reviewadvisor_train"]
class GraphPreprocessor:
    """
    A class to handle the entire data preprocessing pipeline,
    from intermediate viewpoint files to final PyG Data objects.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and config['device'] == 'cuda' else "cpu"
        print(f"GraphPreprocessor using device: {self.device}")

        sbert_model_name = self.config['preprocessing']['sbert_model']
        self.sbert_model = SentenceTransformer(sbert_model_name, device=self.device)

        self._compute_role_pair_map()

    def _compute_role_pair_map(self):
        """Precomputes the mapping from (role1, role2) to a unique integer ID."""
        role_map = self.config['preprocessing']['role_map']
        role_ids = sorted(list(role_map.values()))
        self.pair_to_id = {}
        id_counter = 0
        for r1 in role_ids:
            for r2 in role_ids:
                if (r1, r2) not in self.pair_to_id:
                    self.pair_to_id[(r1, r2)] = id_counter
                    id_counter += 1

    def generate_graph_features(self, typed_viewpoints: List[Dict[str, str]]) -> Optional[Data]:
        """
        Generates a PyG Data object from a list of typed viewpoints.
        """
        num_nodes = len(typed_viewpoints)
        if num_nodes == 0:
            return None

        viewpoint_texts = [item['viewpoint'] for item in typed_viewpoints]
        X = self.sbert_model.encode(viewpoint_texts, convert_to_tensor=True, device=self.device)

        cos_sim = torch.nn.functional.cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=2)
        cos_sim.fill_diagonal_(-1)
        
        k = min(self.config['preprocessing']['knn_k'], num_nodes - 1)
        if k <= 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            top_k = torch.topk(cos_sim, k, dim=1)
            row_indices = torch.arange(num_nodes, device=self.device).view(-1, 1).repeat(1, k)
            edge_index = torch.stack([row_indices.flatten(), top_k.indices.flatten()], dim=0)

        G = nx.Graph()
        G.add_edges_from(edge_index.T.cpu().numpy())
        path_lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=self.config['preprocessing']['max_path_distance']))
        
        max_dist = self.config['preprocessing']['max_path_distance']
        P = torch.full((num_nodes, num_nodes), max_dist, dtype=torch.long)
        for i in range(num_nodes):
            P[i, i] = 0
            if i in path_lengths:
                for j, length in path_lengths[i].items():
                    P[i, j] = length
        
        role_map = self.config['preprocessing']['role_map']
        roles = [role_map[item['role']] for item in typed_viewpoints]
        
        D = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
        for i in range(num_nodes):
            for j in range(num_nodes):
                role_pair = (roles[i], roles[j])
                D[i, j] = self.pair_to_id.get(role_pair, -1)

        return Data(
            x=X.cpu(),
            edge_index=edge_index.cpu(),
            p_matrix=P.cpu(),
            d_matrix=D.cpu(),
        )

    def run(self, dataset_name: str):
        """
        Main function to run the entire preprocessing pipeline for a dataset.
        """
        intermediate_path = f"data/intermediate/{dataset_name}_viewpoints.jsonl"
        processed_dir = os.path.join("data/processed",dataset_name)

        if not os.path.exists(intermediate_path):
            print(f"Error: Intermediate viewpoint file not found at {intermediate_path}")
            print("Please run the API extraction script first.")
            return

        os.makedirs(processed_dir, exist_ok=True)

        with open(intermediate_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Starting graph construction for '{dataset_name}' dataset...")
        decision_map = self.config.get('decision_mapping',[])
        for i, line in enumerate(tqdm(lines, desc=f"Building graphs for {dataset_name}")):
            output_file = os.path.join(processed_dir, f"graph_{i}.pt")
            if os.path.exists(output_file):
                continue

            paper_data = json.loads(line)
            typed_viewpoints = paper_data.get('typed_viewpoints', [])
            
            if not typed_viewpoints:
                continue
            
            graph_data = self.generate_graph_features(typed_viewpoints)
            
            if graph_data:
                decision_str = paper_data.get('decision','Unknown')
                
                label = decision_map.get(decision_str,-1)
                graph_data.y = torch.tensor([label], dtype=torch.long)
                torch.save(graph_data, output_file)

        print(f"Finished graph construction for '{dataset_name}'. Results saved to {processed_dir}")

if __name__ == '__main__':
    config = load_config()
    set_seed(config['seed'])
    preprocessor = GraphPreprocessor(config)
    for file in FILE_LIST:
        preprocessor.run(file)
