import json
import numpy as np
import pandas as pd

go_gene_map = pd.read_csv("go_gene_map.csv").set_index("index")["go_term"].to_dict()
go_terms = pd.read_csv("go_terms.csv").set_index("index")["description"].to_dict()

def save_rna_embeddings_for_neo4j(input_file, embedding_file, output_file):
    df_iter = pd.read_csv(input_file, chunksize=5000)
    embeddings = np.load(embedding_file, mmap_mode='r')

    with open(output_file, "w") as f:
        for chunk in df_iter:
            metadata_cols = ["donor_id", "age", "gender", "structure_name", "age_label", "age_category", "text_prompt"]
            gene_cols = [col for col in chunk.columns if col not in metadata_cols]

            for i, row in chunk.iterrows():
                donor_id_fixed = "_".join(row["donor_id"].replace(".", "_").split("_")[:3])  
                print(f"🔍 Fixed Donor ID: {row['donor_id']} → {donor_id_fixed}") 
                
                for gene in gene_cols:
                    expression_value = pd.to_numeric(row[gene], errors="coerce")

                    if pd.notna(expression_value) and expression_value > 0:
                        go_term = go_gene_map.get(gene, "Unknown GO Term")
                        go_desc = go_terms.get(go_term, "Unknown function")

                        entry = {
                            "donor_id": donor_id_fixed, 
                            "age_label": row["age_label"],
                            "structure_name": row["structure_name"],
                            "gene_symbol": gene,
                            "expression_value": expression_value,
                            "go_description": go_desc,
                            "embedding": embeddings[i].tolist()
                        }
                        f.write(json.dumps(entry) + "\n")

    print(f"RNA-seq JSONL with FIXED Donor IDs saved: {output_file}")


def save_microRNA_embeddings_for_neo4j(input_file, embedding_file, mirna_target_file, output_file):
    df = pd.read_csv(input_file)
    embeddings = np.load(embedding_file, mmap_mode='r')

    # Load miRTarBase target gene mappings
    mirna_targets = pd.read_csv(mirna_target_file, sep=",", header=0)
    mirna_targets = mirna_targets[["miRNA", "Target Gene"]].drop_duplicates()
    mirna_map = mirna_targets.groupby("miRNA")["Target Gene"].apply(list).to_dict()

    if len(df) != len(embeddings):
        print(f"ERROR: Mismatch - {len(df)} rows in CSV, {len(embeddings)} embeddings.")
        return

    metadata_cols = ["donor_id", "age", "gender", "structure_name", "age_category", "age_label", "text_prompt"]
    mirna_cols = [col for col in df.columns if col not in metadata_cols]

    with open(output_file, "w") as f:
        for i, row in df.iterrows():
            donor_id_fixed = "_".join(row["donor_id"].replace(".", "_").split("_")[:3])
            for mirna in mirna_cols:
                expression_value = pd.to_numeric(row[mirna], errors="coerce")

                if pd.notna(expression_value) and expression_value > 0:
                    entry = {
                        "donor_id": donor_id_fixed,  # FIXED Donor ID
                        "age_label": row["age_label"],
                        "structure_name": row["structure_name"],
                        "mirna_symbol": mirna,
                        "mirna_expression": expression_value,
                        "target_genes": mirna_map.get(mirna, []),
                        "embedding": embeddings[i].tolist()
                    }
                    f.write(json.dumps(entry) + "\n")

    print(f"microRNA JSONL file with FIXED Donor IDs saved: {output_file}")


#save_rna_embeddings_for_neo4j("rna_train.csv", "rna_train_embeddings.npy", "rna_train_neo4j.jsonl")
save_rna_embeddings_for_neo4j("rna_test.csv", "rna_test_embeddings.npy", "rna_test_neo4j.jsonl")
save_rna_embeddings_for_neo4j("rna_val.csv", "rna_val_embeddings.npy", "rna_val_neo4j.jsonl")

#save_microRNA_embeddings_for_neo4j("microRNA_train.csv", "microRNA_train_embeddings.npy", "miRTarBase_SE_WR.csv", "microRNA_train_neo4j.jsonl")
save_microRNA_embeddings_for_neo4j("microRNA_test.csv", "microRNA_test_embeddings.npy", "miRTarBase_SE_WR.csv", "microRNA_test_neo4j.jsonl")
save_microRNA_embeddings_for_neo4j("microRNA_val.csv", "microRNA_val_embeddings.npy", "miRTarBase_SE_WR.csv", "microRNA_val_neo4j.jsonl")