import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

rna_train_emb = np.load("rna_train_embeddings.npy")
rna_val_emb = np.load("rna_val_embeddings.npy")
rna_test_emb = np.load("rna_test_embeddings.npy")

micro_train_emb = np.load("microRNA_train_embeddings.npy")
micro_val_emb = np.load("microRNA_val_embeddings.npy")
micro_test_emb = np.load("microRNA_test_embeddings.npy")

rna_train_labels = np.loadtxt("rna_train_embeddings_labels.csv", delimiter=",", skiprows=1, usecols=1)
rna_val_labels = np.loadtxt("rna_val_embeddings_labels.csv", delimiter=",", skiprows=1, usecols=1)
rna_test_labels = np.loadtxt("rna_test_embeddings_labels.csv", delimiter=",", skiprows=1, usecols=1)

micro_train_labels = np.loadtxt("microRNA_train_embeddings_labels.csv", delimiter=",", skiprows=1, usecols=1)
micro_val_labels = np.loadtxt("microRNA_val_embeddings_labels.csv", delimiter=",", skiprows=1, usecols=1)
micro_test_labels = np.loadtxt("microRNA_test_embeddings_labels.csv", delimiter=",", skiprows=1, usecols=1)

clf_rna = LogisticRegression(max_iter=2000)
clf_rna.fit(rna_train_emb, rna_train_labels)

val_preds = clf_rna.predict(rna_val_emb)
test_preds = clf_rna.predict(rna_test_emb)

print(f"RNA-seq Validation Accuracy: {accuracy_score(rna_val_labels, val_preds):.4f}")
print(f"RNA-seq Test Accuracy: {accuracy_score(rna_test_labels, test_preds):.4f}")
print("\nRNA-seq Classification Report:\n", classification_report(rna_test_labels, test_preds))

clf_micro = LogisticRegression(max_iter=2000)
clf_micro.fit(micro_train_emb, micro_train_labels)

val_preds_micro = clf_micro.predict(micro_val_emb)
test_preds_micro = clf_micro.predict(micro_test_emb)

print(f"microRNA Validation Accuracy: {accuracy_score(micro_val_labels, val_preds_micro):.4f}")
print(f"microRNA Test Accuracy: {accuracy_score(micro_test_labels, test_preds_micro):.4f}")
print("\nmicroRNA Classification Report:\n", classification_report(micro_test_labels, test_preds_micro))

print("Classification complete!")