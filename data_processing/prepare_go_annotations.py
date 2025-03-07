import pandas as pd

import os

if not os.path.exists("goa_human.gaf"):
    print("⬇Downloading GO annotations...")
    os.system("wget http://geneontology.org/gene-associations/goa_human.gaf.gz")
    os.system("gunzip goa_human.gaf.gz")

if not os.path.exists("go-basic.obo"):
    print("⬇Downloading GO basic ontology...")
    os.system("wget http://purl.obolibrary.org/obo/go/go-basic.obo -O go-basic.obo")

go_annotations = pd.read_csv(
    "goa_human.gaf", sep="\t", comment="!", header=None, usecols=[2, 4], names=["gene_symbol", "go_term"]
)

go_map = go_annotations.set_index("gene_symbol")["go_term"].to_dict()

def parse_go_descriptions(go_obo_file):
    go_terms = {}
    with open(go_obo_file, "r") as f:
        current_go_id = None
        for line in f:
            line = line.strip()
            if line.startswith("[Term]"):
                current_go_id = None
            elif line.startswith("id: GO:"):
                current_go_id = line.split(": ")[1]
            elif line.startswith("name: ") and current_go_id:
                go_terms[current_go_id] = line.split(": ")[1]
    return go_terms

go_descriptions = parse_go_descriptions("go-basic.obo")

pd.DataFrame.from_dict(go_map, orient="index", columns=["go_term"]).reset_index().to_csv("go_gene_map.csv", index=False)
pd.DataFrame.from_dict(go_descriptions, orient="index", columns=["description"]).reset_index().to_csv("go_terms.csv", index=False)

print("GO Annotations and Descriptions Processed!")