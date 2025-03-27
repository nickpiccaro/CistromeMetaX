import os
import requests
import csv

# Define URLs and file names
factor_urls = [
    "https://guolab.wchscu.cn/AnimalTFDB4_static/download/TF_list_final/Homo_sapiens_TF",
]
factor_filenames = ["Homo_sapiens_TF.csv"]

ontology_urls = [
    "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt",
    "https://github.com/EBISPOT/efo/releases/download/current/efo.owl",
    "http://purl.obolibrary.org/obo/uberon/uberon-full.json"
]
ontology_filenames = ["cellosaurus.txt", "efo.owl", "uberon-full.json"]

# Define directory paths
factor_folder = "data/unparsed_factor_data"
ontology_folder = "data/unparsed_ontology_data"
parsed_folder = "data/parsed_factor_data"

# Create directories if they don't exist
os.makedirs(factor_folder, exist_ok=True)
os.makedirs(ontology_folder, exist_ok=True)
os.makedirs(parsed_folder, exist_ok=True)

def download_file(url, folder, filename):
    """Download a file and save it locally."""
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(folder, filename)
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download {url}")

def download_all():
    """Download all required data files."""
    for url, filename in zip(factor_urls, factor_filenames):
        download_file(url, factor_folder, filename)
    
    for url, filename in zip(ontology_urls, ontology_filenames):
        download_file(url, ontology_folder, filename)

def main():
    """Entry point for the CLI command."""
    print("Downloading all data files...")
    download_all()
    print("Data download complete.")

if __name__ == "__main__":
    main()