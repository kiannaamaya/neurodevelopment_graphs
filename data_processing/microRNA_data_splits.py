import pandas as pd
from sklearn.model_selection import train_test_split

micro_df = pd.read_csv("microRNA_text_prompts.csv")

# Fix Donor IDs: Replace dots with underscores & remove brain region suffixes
micro_df["donor_id"] = micro_df["donor_id"].apply(lambda x: "_".join(x.split("-")[0].split("_")[:3]))

# Load the RNA-seq donor split info
rna_splits = pd.read_csv("rna_donor_splits.csv").set_index("donor_id")["split"].to_dict()

# Get unique donors from microRNA data
micro_donors = set(micro_df["donor_id"].unique())

# Assign splits for shared donors using RNA-seq
train_micro_donors = [d for d in micro_donors if rna_splits.get(d) == "train"]
val_micro_donors = [d for d in micro_donors if rna_splits.get(d) == "val"]
test_micro_donors = [d for d in micro_donors if rna_splits.get(d) == "test"]

# Handle donors that ONLY exist in microRNA
exclusive_micro_donors = list(micro_donors - set(rna_splits.keys()))

if exclusive_micro_donors:
    extra_train, extra_test = train_test_split(exclusive_micro_donors, test_size=0.2, random_state=42)
    extra_train, extra_val = train_test_split(extra_train, test_size=0.1, random_state=42)

    train_micro_donors.extend(extra_train)
    val_micro_donors.extend(extra_val)
    test_micro_donors.extend(extra_test)

# Assign rows based on donor ID
train_micro_df = micro_df[micro_df["donor_id"].isin(train_micro_donors)]
val_micro_df = micro_df[micro_df["donor_id"].isin(val_micro_donors)]
test_micro_df = micro_df[micro_df["donor_id"].isin(test_micro_donors)]

# Save new microRNA datasets
train_micro_df.to_csv("microRNA_train.csv", index=False)
val_micro_df.to_csv("microRNA_val.csv", index=False)
test_micro_df.to_csv("microRNA_test.csv", index=False)

print("microRNA dataset split by donor and saved (with fixed donor IDs)!")