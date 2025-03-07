import os

scripts = [
    #"python data_processing/prepare_go_annotations.py",  # Step 1: Process GO Annotations
    #"python data_processing/rna_text_prompt.py",  # Step 2: Process RNA-seq
    #"python data_processing/microRNA_text_prompt.py",  # Step 3: Process microRNA
    #"python data_processing/rna_data_splits.py",
    #"python data_processing/microRNA_data_splits.py",
    #"python data_processing/confirm_splits.py",
    #"python data_processing/generate_embeddings.py",
    #"python data_processing/train_classifier.py",
    #"python data_processing/test_classifier.py",
    #"python data_processing/prepare_neo4j_json.py",
    "python neo4j_process/neo4j_insert_rna.py",
    "python neo4j_process/neo4j_insert_microRNA.py"
    #"python neo4j_process/export_neo4j_data.py",
    #"python gnns/gnn_processing.py",
    #"python gnns/gnn_model.py",
]

for script in scripts:
    os.system(script)