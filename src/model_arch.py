# # src/model_arch.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import MessagePassing, global_mean_pool
# from torch_geometric.utils import softmax
# from typing import Dict, Any

# class SAGA_Layer(MessagePassing):
#     """
#     Structurally-Aware Graph Attention (SAGA) Layer.
#     This layer computes attention scores by incorporating structural biases
#     from path distances (P) and node roles (D).
#     """
#     def __init__(self, in_dim: int, out_dim: int, config: Dict[str, Any]):
#         super(SAGA_Layer, self).__init__(node_dim=0, aggr='add')
        
#         self.in_dim = in_dim
#         self.out_dim = out_dim
        
#         # Linear projections for Query, Key, Value
#         self.w_q = nn.Linear(in_dim, out_dim, bias=False)
#         self.w_k = nn.Linear(in_dim, out_dim, bias=False)
#         self.w_v = nn.Linear(in_dim, out_dim, bias=False)
        
#         # Learnable embeddings for structural biases
#         max_dist = config['preprocessing']['max_path_distance']
#         num_roles = len(config['preprocessing']['role_map'])
#         num_role_pairs = num_roles * num_roles
        
#         self.f_p = nn.Embedding(max_dist + 1, 1) # +1 to handle the max_dist value
#         self.f_d = nn.Embedding(num_role_pairs, 1)

#         # Feed-forward network
#         self.ffn = nn.Sequential(
#             nn.Linear(out_dim, out_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(config['model']['dropout']),
#             nn.Linear(out_dim * 2, out_dim)
#         )
#         self.ln1 = nn.LayerNorm(out_dim)
#         self.ln2 = nn.LayerNorm(out_dim)
#         self.dropout = nn.Dropout(config['model']['dropout'])

#     def forward(self, x, edge_index, p_matrix, d_matrix):
#         # Apply layer normalization before attention (Pre-LN)
#         x_norm = self.ln1(x)
        
#         # Project to Q, K, V
#         q = self.w_q(x_norm)
#         k = self.w_k(x_norm)
#         v = self.w_v(x_norm)
        
#         # Propagate messages (computes attention and aggregates values)
#         # We pass K and V to the message passing mechanism
#         out = self.propagate(edge_index, q=q, k=k, v=v, p_matrix=p_matrix, d_matrix=d_matrix)
        
#         # Residual connection and dropout
#         x = x + self.dropout(out)
        
#         # Second residual block with FFN
#         out = self.ffn(self.ln2(x))
#         x = x + self.dropout(out)
        
#         return x

#     def message(self, q_i, k_j, v_j, edge_index, p_matrix, d_matrix):
#         # q_i: Query of the target node
#         # k_j, v_j: Key and Value of the source node
        
#         # Get structural information for the edge
#         row, col = edge_index
#         p_ij = p_matrix[row, col].view(-1)
#         d_ij = d_matrix[row, col].view(-1)
        
#         # Get learnable biases from embeddings
#         bias_p = self.f_p(p_ij).view(-1, 1)
#         bias_d = self.f_d(d_ij).view(-1, 1)
        
#         # Compute attention score
#         scale = self.out_dim ** -0.5
#         score = (q_i * k_j).sum(dim=-1, keepdim=True) * scale
        
#         # Add structural biases to the score
#         attention_score = score + bias_p + bias_d
        
#         # Attention weights are computed in propagate() via softmax
#         self.alpha = attention_score
        
#         # Return weighted values
#         return v_j * softmax(self.alpha, edge_index[0], num_nodes=q_i.size(0)).view(-1, 1)

# class GraphEvalGT(nn.Module):
#     """
#     The main GraphEval-GT model, which stacks SAGA layers and adds a final classifier.
#     """
#     def __init__(self, config: Dict[str, Any]):
#         super(GraphEvalGT, self).__init__()
        
#         in_dim = config['model']['in_dim']
#         hidden_dim = config['model']['hidden_dim']
#         out_dim = config['model']['out_dim']
#         num_layers = config['model']['num_layers']
        
#         self.input_proj = nn.Linear(in_dim, hidden_dim)
        
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.layers.append(SAGA_Layer(hidden_dim, hidden_dim, config))
            
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(config['model']['dropout']),
#             nn.Linear(hidden_dim // 2, out_dim)
#         )

#     def forward(self, data):
#         x, edge_index, p_matrix, d_matrix, batch = \
#             data.x, data.edge_index, data.p_matrix, data.d_matrix, data.batch

#         x = self.input_proj(x)
        
#         for layer in self.layers:
#             x = layer(x, edge_index, p_matrix, d_matrix)
            
#         # Graph-level readout
#         graph_embedding = global_mean_pool(x, batch)
        
#         # Final classification
#         logits = self.classifier(graph_embedding)
        
#         return logits

# # ==============================================================================
# # Self-Contained Test Block
# # ==============================================================================
# if __name__ == '__main__':
#     print("Running self-contained test for model_arch.py...")

#     # --- 1. Setup: Load a dummy config and create dummy data ---
#     from config_loader import load_config
#     # from torch_geometric.data import Data
    
#     config = load_config()
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     num_nodes = 10
#     in_dim = config['model']['in_dim']
#     hidden_dim = config['model']['hidden_dim']
#     num_classes = config['model']['out_dim']
    
#     # Dummy graph data
#     # dummy_x = torch.randn(num_nodes, in_dim, device=device)
#     # dummy_edge_index = torch.randint(0, num_nodes, (2, 20), device=device)
#     # dummy_p_matrix = torch.randint(0, config['preprocessing']['max_path_distance'], (num_nodes, num_nodes), device=device)
#     # dummy_d_matrix = torch.randint(0, 25, (num_nodes, num_nodes), device=device)
#     # dummy_batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

#     # dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, 
#     #                   p_matrix=dummy_p_matrix, d_matrix=dummy_d_matrix, 
#     #                   batch=dummy_batch).to(device)

#     # --- 2. Test SAGA_Layer ---
#     print("\n--- Testing SAGA_Layer ---")
#     try:
#         # **修正**: SAGA_Layer的输入维度应该是hidden_dim
#         dummy_x_for_layer = torch.randn(num_nodes, hidden_dim, device=device)
#         dummy_edge_index = torch.randint(0, num_nodes, (2, 20), device=device)
#         dummy_p_matrix = torch.randint(0, config['preprocessing']['max_path_distance'], (num_nodes, num_nodes), device=device)
#         dummy_d_matrix = torch.randint(0, 25, (num_nodes, num_nodes), device=device)
        
#         saga_layer = SAGA_Layer(hidden_dim, config).to(device)
#         output = saga_layer(dummy_x_for_layer, dummy_edge_index, dummy_p_matrix, dummy_d_matrix)
        
#         assert output.shape == (num_nodes, hidden_dim), f"SAGA_Layer output shape mismatch: {output.shape}"
#         print("SAGA_Layer forward pass successful.")
#     except Exception as e:
#         print(f"SAGA_Layer test failed: {e}")
#         # Re-raise the exception to make it clear the test failed
#         raise e
        
#     # --- 3. Test GraphEvalGT Model ---
#     print("\n--- Testing GraphEvalGT Model ---")
#     try:
#         model = GraphEvalGT(config).to(device)
#         logits = model(dummy_data)
#         assert logits.shape == (1, num_classes), f"GraphEvalGT output shape mismatch: {logits.shape}"
#         print("GraphEvalGT forward pass successful.")
#     except Exception as e:
#         print(f"GraphEvalGT test failed: {e}")
        
#     print("\n[SUCCESS] All tests in model_arch.py passed!")
# src/model_arch.py (修正版)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool,GCNConv
from torch_geometric.data import Data 
from torch_geometric.utils import softmax
from typing import Dict, Any
class SimpleGCN(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(SimpleGCN, self).__init__()
        
        in_dim = config['model']['in_dim']
        hidden_dim = config['model']['hidden_dim']
        out_dim = config['model']['out_dim']
        num_layers = config['model']['num_layers']
        dropout = config['model']['dropout']

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        graph_embedding = global_mean_pool(x, batch)
        logits = self.classifier(graph_embedding)
        
        return logits
class SAGA_Layer(MessagePassing):
    """
    Structurally-Aware Graph Attention (SAGA) Layer.
    This layer computes attention scores by incorporating structural biases
    from path distances (P) and node roles (D).
    """
    def __init__(self, hidden_dim: int, config: Dict[str, Any]):
        super(SAGA_Layer, self).__init__(node_dim=0, aggr='add')
        
        self.hidden_dim = hidden_dim
        
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        max_dist = config['preprocessing']['max_path_distance']
        num_roles = len(config['preprocessing']['role_map'])
        num_role_pairs = num_roles * num_roles
        
        self.f_p = nn.Embedding(max_dist + 1, 1)
        self.f_d = nn.Embedding(num_role_pairs, 1)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(), # GELU is often slightly better than ReLU in Transformers
            nn.Dropout(config['model']['dropout']),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(config['model']['dropout'])

    def forward(self, x, edge_index, p_matrix, d_matrix):
        x_norm = self.ln1(x)
        
        q = self.w_q(x_norm)
        k = self.w_k(x_norm)
        v = self.w_v(x_norm)
        
        out = self.propagate(edge_index, q=q, k=k, v=v, p_matrix=p_matrix, d_matrix=d_matrix)
        
        x = x + self.dropout(out)
        
        out = self.ffn(self.ln2(x))
        x = x + self.dropout(out)
        
        return x

    def message(self, q_i, k_j, v_j, edge_index, index, size_i, p_matrix, d_matrix):
        
        row, col = edge_index
        
        p_ij = p_matrix[row, col]
        d_ij = d_matrix[row, col]
        
        bias_p = self.f_p(p_ij).view(-1, 1)
        bias_d = self.f_d(d_ij).view(-1, 1)
        
        scale = self.hidden_dim ** -0.5
        score = (q_i * k_j).sum(dim=-1, keepdim=True) * scale
        
        attention_score = score + bias_p + bias_d
        
        alpha = softmax(attention_score, index, num_nodes=size_i)
        
        return v_j * alpha

class GraphEvalGT(nn.Module):
    """
    The main GraphEval-GT model, which stacks SAGA layers and adds a final classifier.
    """
    def __init__(self, config: Dict[str, Any]):
        super(GraphEvalGT, self).__init__()
        
        in_dim = config['model']['in_dim']
        hidden_dim = config['model']['hidden_dim']
        out_dim = config['model']['out_dim']
        num_layers = config['model']['num_layers']
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SAGA_Layer(hidden_dim, config))
            
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, data):
        x, edge_index, p_matrix, d_matrix, batch = \
            data.x, data.edge_index, data.p_matrix, data.d_matrix, data.batch

        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x, edge_index, p_matrix, d_matrix)
            
        graph_embedding = global_mean_pool(x, batch)
        
        logits = self.classifier(graph_embedding)
        
        return logits

