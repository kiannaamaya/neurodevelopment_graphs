import pandas as pd
from sklearn.model_selection import train_test_split

# Load RNA dataset
rna_df = pd.read_csv("rna_text_prompts.csv")

# Fix Donor IDs: Replace dots with underscores & remove brain region suffixes
rna_df["donor_id"] = rna_df["donor_id"].apply(lambda x: "_".join(x.replace(".", "_").split("_")[:3]))

# Group by donor ID and get unique donors
unique_donors = rna_df["donor_id"].unique()

# Split donors into train, validation, and test
train_donors, test_donors = train_test_split(unique_donors, test_size=0.2, random_state=42)
train_donors, val_donors = train_test_split(train_donors, test_size=0.1, random_state=42)

# Save donor splits for microRNA script to use later
donor_splits = pd.DataFrame({
    "donor_id": list(unique_donors),
    "split": ["train" if d in train_donors else "val" if d in val_donors else "test" for d in unique_donors]
})
donor_splits.to_csv("rna_donor_splits.csv", index=False)

# Assign rows to splits based on fixed donor ID
train_rna_df = rna_df[rna_df["donor_id"].isin(train_donors)]
val_rna_df = rna_df[rna_df["donor_id"].isin(val_donors)]
test_rna_df = rna_df[rna_df["donor_id"].isin(test_donors)]

# Save new datasets
train_rna_df.to_csv("rna_train.csv", index=False)
val_rna_df.to_csv("rna_val.csv", index=False)
test_rna_df.to_csv("rna_test.csv", index=False)

print("RNA dataset split by donor and saved (with fixed donor IDs)!")
print("RNA donor splits saved to rna_donor_splits.csv!")