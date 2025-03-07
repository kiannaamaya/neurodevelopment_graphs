import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("pritamdeka/S-BioBERT-snli-multinli-stsb")

def generate_and_save_embeddings(input_file, output_file):
    """Generates embeddings for text prompts and saves them with age labels."""
    df = pd.read_csv(input_file)

    embeddings = model.encode(df["text_prompt"].tolist(), convert_to_numpy=True)

    np.save(output_file, embeddings)

    df[["donor_id", "age_label"]].to_csv(output_file.replace(".npy", "_labels.csv"), index=False)

    print(f"Saved embeddings: {output_file} ({embeddings.shape})")
    return embeddings, df["age_label"].values

rna_train_emb, rna_train_labels = generate_and_save_embeddings("rna_train.csv", "rna_train_embeddings.npy")
rna_val_emb, rna_val_labels = generate_and_save_embeddings("rna_val.csv", "rna_val_embeddings.npy")
rna_test_emb, rna_test_labels = generate_and_save_embeddings("rna_test.csv", "rna_test_embeddings.npy")

micro_train_emb, micro_train_labels = generate_and_save_embeddings("microRNA_train.csv", "microRNA_train_embeddings.npy")
micro_val_emb, micro_val_labels = generate_and_save_embeddings("microRNA_val.csv", "microRNA_val_embeddings.npy")
micro_test_emb, micro_test_labels = generate_and_save_embeddings("microRNA_test.csv", "microRNA_test_embeddings.npy")
