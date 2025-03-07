import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

print("Loading Data...")

# Load Nodes
rna_genes = pd.read_csv("rna_genes.csv")
mirna_genes = pd.read_csv("microrna_genes.csv")
donors = pd.read_csv("donors.csv")

# Load Edges
rna_edges = pd.read_csv("expresses_rna_edges.csv")
mirna_edges = pd.read_csv("expresses_microrna_edges.csv")
mirna_target_edges = pd.read_csv("microrna_target_edges.csv")

# Convert Edge Columns to Integers
rna_edges["source"] = pd.to_numeric(rna_edges["source"], errors="coerce").astype("Int64")
rna_edges["target"] = pd.to_numeric(rna_edges["target"], errors="coerce").astype("Int64")
rna_edges.dropna(inplace=True)  # Remove NaNs
rna_edges = rna_edges.astype(int)

mirna_edges["source"] = pd.to_numeric(mirna_edges["source"], errors="coerce").astype("Int64")
mirna_edges["target"] = pd.to_numeric(mirna_edges["target"], errors="coerce").astype("Int64")
mirna_edges.dropna(inplace=True)
mirna_edges = mirna_edges.astype(int)

mirna_target_edges["source"] = pd.to_numeric(mirna_target_edges["source"], errors="coerce").astype("Int64")
mirna_target_edges["target"] = pd.to_numeric(mirna_target_edges["target"], errors="coerce").astype("Int64")
mirna_target_edges.dropna(inplace=True)
mirna_target_edges = mirna_target_edges.astype(int)

# Donors → Encode age labels for classification
le = LabelEncoder()
donors["age_label"] = le.fit_transform(donors["age_label"])

# Create edge index for RNA Expression
rna_edge_index = torch.tensor([rna_edges["source"].values, rna_edges["target"].values], dtype=torch.long)

# Create edge index for microRNA Expression
mirna_edge_index = torch.tensor([mirna_edges["source"].values, mirna_edges["target"].values], dtype=torch.long)

# Convert Labels to Tensor
age_labels = torch.tensor(donors["age_label"].values, dtype=torch.long)

rna_data = Data(edge_index=rna_edge_index, y=age_labels)

mirna_data = Data(edge_index=mirna_edge_index, y=age_labels)

print("RNA & microRNA Graph Data Ready for GNN Training!")

torch.save(rna_data, "rna_graph_data.pt")
torch.save(mirna_data, "mirna_graph_data.pt")