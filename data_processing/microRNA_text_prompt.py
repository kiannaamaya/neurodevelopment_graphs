import pandas as pd

microRNA_df = pd.read_csv("microRNA_final.csv")

mirna_targets = pd.read_csv("miRTarBase_SE_WR.csv", sep=",", header=0)

print("Columns in miRTarBase file:", mirna_targets.columns.tolist())

mirna_targets = mirna_targets[["miRNA", "Target Gene"]]

mirna_target_map = mirna_targets.groupby("miRNA")["Target Gene"].apply(list).to_dict()

metadata_cols = ["donor_id", "age", "gender", "structure_acronym", "structure_name", "age_category", "age_label"]
mirna_cols = [col for col in microRNA_df.columns if col not in metadata_cols]

def generate_microRNA_text_prompt(row):
    mirna_text = []
    metadata = f"Donor {row['donor_id']} with gender {row['gender']}. Brain region is {row['structure_name']}."

    for mirna in mirna_cols:
        expression_value = pd.to_numeric(row[mirna], errors="coerce")
        if pd.notna(expression_value) and expression_value > 0:
            target_genes = mirna_target_map.get(mirna, [])
            gene_list = ", ".join(target_genes) if target_genes else "No known targets"
            mirna_text.append(f"{mirna} expression is {expression_value:.2f}. Target genes include {gene_list}.")

    return metadata + " " + " ".join(mirna_text)

microRNA_df["text_prompt"] = microRNA_df.apply(generate_microRNA_text_prompt, axis=1)

microRNA_df[metadata_cols + mirna_cols + ["text_prompt"]].to_csv("microRNA_text_prompts.csv", index=False)

print("microRNA text prompts saved with only target gene names!")