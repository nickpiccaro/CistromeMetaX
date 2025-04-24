import os
import csv
import subprocess
import requests
from pathlib import Path


def get_data_dir():
    """Returns the path to the data directory within the installed package."""
    return Path(__file__).parent / "data"


def download_file(url, folder, filename, as_text=False):
    """Downloads a file from a given URL and saves it in the specified folder."""
    file_path = folder / filename
    try:
        if filename == "Homo_sapiens_TF.csv":  # Use curl for this specific file
            subprocess.run(
                [
                    "curl",
                    "-o",
                    file_path,
                    "-A",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    url,
                ],
                check=True,
            )
            print(f"Downloaded and saved {filename} in {folder}")
        else:  # Use requests for other files
            response = requests.get(
                url, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True
            )  # Allow redirects in case of issues
            if response.status_code == 200:
                mode = "w" if as_text else "wb"
                encoding = "utf-8" if as_text else None  # Only set encoding for text mode

                with open(file_path, mode, encoding=encoding) as file:
                    if as_text:
                        file.write(response.text)  # Save as plain text
                    else:
                        file.write(response.content)  # Save as binary data

                print(f"Downloaded and saved {filename} in {folder}")
            else:
                print(f"Failed to download {url} (Status code: {response.status_code})")
    except (requests.exceptions.RequestException, subprocess.CalledProcessError) as e:
        print(f"Error downloading {url}: {e}")
    except FileNotFoundError:
        print("Error: curl command not found. Please ensure curl is installed and in your system's PATH.")

def fetch_chromatin_remodelers_and_synonyms(output_csv):
    # Base URLs
    base_url = "https://maayanlab.cloud/Harmonizome"
    chromatin_remodelers_url = f"{base_url}/api/1.0/gene_set/chromatin+remodeling/GO+Biological+Process+Annotations+2023"
    
    try:
        # Step 1: Fetch all chromatin remodelers
        response = requests.get(chromatin_remodelers_url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        associations = data.get("associations", [])
        
        # Step 2: Prepare a list to hold the results
        chromatin_data = []
        
        for association in associations:
            gene_symbol = association["gene"]["symbol"]
            gene_url = f"{base_url}{association['gene']['href']}"
            
            # Step 3: Fetch individual gene details
            gene_response = requests.get(gene_url)
            gene_response.raise_for_status()
            gene_data = gene_response.json()
            
            synonyms = gene_data.get("synonyms", [])
            chromatin_data.append({
                "chromatin_remodeler": gene_symbol,
                "synonyms": ", ".join(synonyms) if synonyms else ""
            })
        
        # Step 4: Write data to CSV
        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["chromatin_remodeler", "synonyms"])
            writer.writeheader()
            writer.writerows(chromatin_data)
        
        print(f"Data successfully saved to {output_csv}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def install_data():
    """Downloads required data and organizes it into directories."""
    print("GEOMetaX | Installing data...")
    data_dir = get_data_dir()
    os.makedirs(data_dir / "unparsed_factor_data", exist_ok=True)
    os.makedirs(data_dir / "unparsed_ontology_data", exist_ok=True)
    os.makedirs(data_dir / "parsed_factor_data", exist_ok=True)
    os.makedirs(data_dir / "parsed_ontology_data", exist_ok=True)

    factor_urls = [
        (
            "https://ftp.ncbi.nih.gov/gene/DATA/gene_info.gz",
            "gene_info.gz",
            False,
        ),  # Binary file
        (
            "https://guolab.wchscu.cn/AnimalTFDB4_static/download/TF_list_final/Homo_sapiens_TF",
            "Homo_sapiens_TF.csv",
            True,
        ),  # Text file
    ]

    ontology_urls = [
        (
            "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt",
            "cellosaurus.txt",
            True,
        ),
        (
            "https://github.com/EBISPOT/efo/releases/download/current/efo.owl",
            "efo.owl",
            True,
        ),
        (
            "http://purl.obolibrary.org/obo/uberon/uberon-full.json",
            "uberon-full.json",
            True,
        ),
    ]

    # Download factor-related data
    for url, filename, as_text in factor_urls:
        download_file(url, data_dir / "unparsed_factor_data", filename, as_text)
    fetch_chromatin_remodelers_and_synonyms(data_dir / "unparsed_factor_data/Homo_sapiens_CR.csv")
    # Download ontology data
    for url, filename, as_text in ontology_urls:
        download_file(url, data_dir / "unparsed_ontology_data", filename, as_text)


if __name__ == "__main__":
    install_data()