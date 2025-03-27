import os
import csv

def list_downloaded_files():
    """List downloaded files."""
    factor_folder = "data/unparsed_factor_data"
    ontology_folder = "data/unparsed_ontology_data"

    print("\nFactor Data Files:")
    for f in os.listdir(factor_folder):
        print(f"- {f}")
    
    print("\nOntology Data Files:")
    for f in os.listdir(ontology_folder):
        print(f"- {f}")

def process_data():
    """Dummy function to process data."""
    factor_file = "data/unparsed_factor_data/Homo_sapiens_TF.csv"
    output_file = "data/parsed_factor_data/processed_factors.csv"

    if not os.path.exists(factor_file):
        print("Factor file not found. Please run `fetch_data` first.")
        return

    with open(factor_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        lines = list(reader)

    # Example: Extract first 10 rows
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(lines[:10])

    print(f"Processed data saved in {output_file}")

if __name__ == "__main__":
    process_data()