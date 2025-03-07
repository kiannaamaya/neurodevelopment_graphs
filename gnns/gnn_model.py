import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.serialization import add_safe_globals
from torch_geometric.data import Data

add_safe_globals([Data])

rna_data = torch.load("rna_graph_data.pt", weights_only=False)
mirna_data = torch.load("mirna_graph_data.pt", weights_only=False)

print("RNA & microRNA Graph Data Loaded Successfully!")

rna_data.num_nodes = rna_data.edge_index.max().item() + 1
rna_data.x = torch.ones((rna_data.num_nodes, 1))  # Dummy feature if none exist

mirna_data.num_nodes = mirna_data.edge_index.max().item() + 1
mirna_data.x = torch.ones((mirna_data.num_nodes, 1))

# Fix `y` so it only applies to Donor Nodes**
num_donors = 28  # Number of donors
rna_data.y = torch.full((rna_data.num_nodes,), -1, dtype=torch.long)
rna_data.y[:num_donors] = torch.arange(num_donors)

mirna_data.y = torch.full((mirna_data.num_nodes,), -1, dtype=torch.long)
mirna_data.y[:num_donors] = torch.arange(num_donors)

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize Model for RNA
rna_model = GNN(input_dim=1, hidden_dim=32, output_dim=num_donors)
rna_optimizer = torch.optim.Adam(rna_model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Initialize Model for microRNA
mirna_model = GNN(input_dim=1, hidden_dim=32, output_dim=num_donors)
mirna_optimizer = torch.optim.Adam(mirna_model.parameters(), lr=0.01)

print("Training RNA GNN...")
for epoch in range(100):
    rna_model.train()
    rna_optimizer.zero_grad()
    out = rna_model(rna_data.x, rna_data.edge_index)

    # Ensure `donor_mask` Matches Shape
    donor_mask = (rna_data.y >= 0)  # Select donor nodes
    loss = loss_fn(out[donor_mask], rna_data.y[donor_mask])

    loss.backward()
    rna_optimizer.step()
    if epoch % 10 == 0:
        print(f"RNA Epoch {epoch}: Loss = {loss.item()}")

print("Training microRNA GNN...")
for epoch in range(100):
    mirna_model.train()
    mirna_optimizer.zero_grad()
    out = mirna_model(mirna_data.x, mirna_data.edge_index)

    # Ensure `donor_mask` Matches Shape
    donor_mask = (mirna_data.y >= 0)
    loss = loss_fn(out[donor_mask], mirna_data.y[donor_mask])

    loss.backward()
    mirna_optimizer.step()
    if epoch % 10 == 0:
        print(f"microRNA Epoch {epoch}: Loss = {loss.item()}")

print("RNA & microRNA GNN Training Complete!")

rna_model.eval()
mirna_model.eval()

donor_mask_rna = (rna_data.y >= 0)
donor_mask_mirna = (mirna_data.y >= 0)

rna_preds = rna_model(rna_data.x, rna_data.edge_index).argmax(dim=1)
mirna_preds = mirna_model(mirna_data.x, mirna_data.edge_index).argmax(dim=1)

rna_accuracy = (rna_preds[donor_mask_rna] == rna_data.y[donor_mask_rna]).sum().item() / donor_mask_rna.sum().item()
mirna_accuracy = (mirna_preds[donor_mask_mirna] == mirna_data.y[donor_mask_mirna]).sum().item() / donor_mask_mirna.sum().item()

print(f"RNA Model Accuracy: {rna_accuracy * 100:.2f}%")
print(f"microRNA Model Accuracy: {mirna_accuracy * 100:.2f}%")