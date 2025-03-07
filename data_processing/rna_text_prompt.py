import pandas as pd

rna_df = pd.read_csv("rna_final.csv")
go_gene_map = pd.read_csv("go_gene_map.csv").set_index("index")["go_term"].to_dict()
go_terms = pd.read_csv("go_terms.csv").set_index("index")["description"].to_dict()

metadata_cols = ["donor_id", "age", "gender", "structure_name", "age_label", "age_category", "text_prompt"]
gene_cols = [col for col in rna_df.columns if col not in metadata_cols]

def generate_text_prompt(row):
    genes_text = []
    metadata = f"Donor {row['donor_id']} with gender {row['gender']}. Brain region is {row['structure_name']}."

    for gene in row.index:
        if gene in ["donor_id", "age", "gender", "structure_name", "age_label", "age_category"]:
            continue  # Skip metadata columns

        expression_value = pd.to_numeric(row[gene], errors="coerce")
        if pd.notna(expression_value) and expression_value > 0:
            go_term = go_gene_map.get(gene, "Unknown GO Term")
            go_desc = go_terms.get(go_term, "Unknown function")
            genes_text.append(f"{gene} expression is {expression_value:.2f}. It is involved in {go_desc}.")

    return metadata + " " + " ".join(genes_text)

rna_df["text_prompt"] = rna_df.apply(generate_text_prompt, axis=1)

rna_df[["donor_id", "age", "gender", "structure_name", "age_label"] + gene_cols + ["text_prompt"]].to_csv("rna_text_prompts.csv", index=False)

print("RNA-seq data successfully converted into structured text for LLM).")
