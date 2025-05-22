import json
from rdflib import Graph
import gzip
import csv
import os
import shutil
from pathlib import Path
import numpy as np
import re
import json


def get_data_dir():
    """Returns the path to the data directory within the installed package."""
    return Path(__file__).parent / "data"

def remove_spaces(text):
    return re.sub(r'\s+', '', text)

def clean_input(input_string):
    """
    Cleans the input string by converting it to lowercase and removing spaces, 
    newlines, and non-alphanumeric characters. Returns an empty string if input is None.

    Args:
        input_string (str): The string to be cleaned.

    Returns:
        str: The cleaned string.
    """
    if input_string is None:
        return ''
    # Convert the input string to lowercase, remove spaces, newlines, and symbols
    cleaned_string = re.sub(r'[^\w]', '', input_string.lower())
    return cleaned_string


def process_string(s, remove=False):
        if not s:  # Handles None and empty string cases
            return None
        if remove:
            words_to_remove = ["cell", "line", "primary", "the", "of", "and", 
                           "or", "but", "tissue", "cells", "human", "for", 
                           "organ", "region", "system", "sample", "assay", 
                           "disease", "to","measurement", "down", "up", "part",
                           "layer", "membrane", "area", "to", "like", "related",
                           "type", "with", "amount", "containing"]
        else:
            words_to_remove = []
        
        words = s.split()
        cleaned_words = [clean_input(word) for word in words]
        filtered_words = [word for word in cleaned_words if word not in words_to_remove and word != '']
        return " ".join(filtered_words)

def move_file(source_path, destination_folder):
    """
    Moves a file from source_path to destination_folder.

    Args:
        source_path (str or Path): Path to the file to move.
        destination_folder (str or Path): Destination folder to move the file into.
    """
    source = Path(source_path)
    destination = Path(destination_folder)

    # Ensure the destination folder exists
    destination.mkdir(parents=True, exist_ok=True)

    # Create the full destination path (folder + filename)
    destination_path = destination / source.name

    # Move the file
    shutil.move(str(source), str(destination_path))
    print(f"Moved file to: {destination_path}")

def delete_folder(folder_path):
    """
    Deletes a folder and all its contents.

    Args:
        folder_path (str or Path): Path to the folder to delete.
    """
    folder = Path(folder_path)

    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)
        print(f"Deleted folder: {folder}")
    else:
        print(f"Folder not found: {folder}")

def clean_input_fuzzy(input_string):
    """
    Cleans the input string by converting it to lowercase and removing symbols,
    but preserving spaces.
    """
    if input_string is None:
        return ''
    return re.sub(r'[^\w\s]', '', input_string.lower())

def build_index_fuzzy(data, ontology_type):
    """Builds an index for EFO and Uberon labels and synonyms, allowing multiple matches per term."""
    index = {}
    for item in data:
        normalized_label = clean_input_fuzzy(item.get("label", ""))
        ontology_entry = {"ontology_accession": item.get("ontology_id", ""), "ontology_type": ontology_type+"fuzzy", "official_term": item.get("label", "")}

        if normalized_label in index:
            index[normalized_label].append(ontology_entry)
        else:
            index[normalized_label] = [ontology_entry]

        for synonym in item.get("exact_synonyms", []) + item.get("related_synonyms", []) + item.get("broad_synonyms", []):
            normalized_synonym = clean_input_fuzzy(synonym)
            if normalized_synonym in index:
                index[normalized_synonym].append(ontology_entry)
            else:
                index[normalized_synonym] = [ontology_entry]
    return index

def build_index_cellosaurus_fuzzy(cellosaurus_data):
    """Builds an index for Cellosaurus data, allowing multiple matches per term."""
    index = {}
    for item in cellosaurus_data:
        ontology_entry = {"ontology_accession": item.get("AC", ""), "ontology_type": "Cellosaurus", "term": item.get("ID", ""), "term_identity": "cell_line", "official_term": item.get("ID", "")}

        for key in ["ID", "AC"]:
            normalized_key = clean_input_fuzzy(item.get(key, ""))
            if normalized_key:
                if normalized_key in index:
                    index[normalized_key].append(ontology_entry)
                else:
                    index[normalized_key] = [ontology_entry]

        for synonym in item.get("SY", []):
            normalized_synonym = clean_input_fuzzy(synonym)
            if normalized_synonym in index:
                index[normalized_synonym].append(ontology_entry)
            else:
                index[normalized_synonym] = [ontology_entry]

    return index

def build_index_cellosaurus(cellosaurus_data, remove_words=False, spaces=True):
    """Builds an index for Cellosaurus data, allowing multiple matches per term."""
    index = {}
    for item in cellosaurus_data:
        ontology_entry = {
            "ontology_accession": item.get("AC", ""),
            "ontology_type": "Cellosaurus",
            "term": item.get("ID", ""),
            "term_identity": "cell_line",
            "official_term": item.get("ID", "")
        }

        # Handle primary ID
        term = item.get("ID")
        if isinstance(term, str):
            normalized_key = process_string(term, remove=remove_words)
            if not spaces:
                normalized_key = remove_spaces(normalized_key)
            if normalized_key:
                index.setdefault(normalized_key, []).append(ontology_entry)

        # Handle synonyms
        for synonym in item.get("SY", []):
            if isinstance(synonym, str):
                normalized_synonym = process_string(synonym, remove=remove_words)
                if not spaces:
                    normalized_synonym = remove_spaces(normalized_synonym)
                if normalized_synonym:
                    index.setdefault(normalized_synonym, []).append(ontology_entry)

    return index


def process_cellosaurus_file(input_file_path, output_file_path, output_file_path_reduce, output_file_path_fuzzy):
    """
    Processes a Cellosaurus text file, filters human data objects, extracts desired fields, and saves as JSON.

    Args:
        input_file_path (str): Path to the input text file.
        output_file_path (str): Path to save the resulting JSON file.
    """
    desired_fields = {"ID", "AC", "SY", "OX", "CA"}

    def parse_entry(entry):
        parsed_data = {}
        for line in entry.strip().split("\n"):
            if line[:2] in desired_fields:
                key = line[:2]
                value = line[5:].strip()
                if key == "SY":
                    parsed_data[key] = [syn.strip() for syn in value.split(";") if syn.strip()]
                else:
                    parsed_data[key] = value
        return parsed_data

    with open(input_file_path, "r") as file:
        data = file.read()

    entries = data.split("//")
    filtered_data = [
        parse_entry(entry)
        for entry in entries
        if "OX   NCBI_TaxID=9606; ! Homo sapiens (Human)" in entry and parse_entry(entry)
    ]

    cellosaurus_fuzzy_index = build_index_cellosaurus_fuzzy(filtered_data)
    cellosaurus_reduce_index = build_index_cellosaurus(filtered_data, remove_words=True, spaces=False)
    cellosaurus_index = build_index_cellosaurus(filtered_data, remove_words=False, spaces=False)

    with open(output_file_path, "w") as f:
        json.dump(cellosaurus_index, f, indent=2)
    with open(output_file_path_reduce, "w") as f:
        json.dump(cellosaurus_reduce_index, f, indent=2)
    with open(output_file_path_fuzzy, "w") as f:
        json.dump(cellosaurus_fuzzy_index, f, separators=(",", ":"))

    print(f"Processed Cellosaurus data has been saved to {output_file_path}")

def process_efo_owl(input_file_path, output_file_path, output_file_path_reduce, output_file_path_fuzzy):
    """Parses the EFO OWL file and outputs the parsed data in JSON format."""
    from rdflib import Graph
    import json

    g = Graph()
    g.parse(input_file_path, format='xml')
    json_data = json.loads(g.serialize(format='json-ld'))

    parsed_classes = []
    for entry in json_data:
        if "@type" in entry and "http://www.w3.org/2002/07/owl#Class" in entry["@type"]:
            label_data = entry.get("http://www.w3.org/2000/01/rdf-schema#label", [{}])
            label = label_data[0].get("@value") if label_data and isinstance(label_data, list) else None

            if not label:
                continue

            exact_synonyms = [syn["@value"] for syn in entry.get("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym", []) if "@value" in syn]
            related_synonyms = [syn["@value"] for syn in entry.get("http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym", []) if "@value" in syn]
            broad_synonyms = [syn["@value"] for syn in entry.get("http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym", []) if "@value" in syn]

            reference = entry.get("@id")
            ontology_id = reference.split('/')[-1] if reference else None
            subclass_of = [sub["@id"] for sub in entry.get("http://www.w3.org/2000/01/rdf-schema#subClassOf", []) if "@id" in sub]

            parsed_classes.append({
                "label": label,
                "exact_synonyms": exact_synonyms,
                "related_synonyms": related_synonyms,
                "broad_synonyms": broad_synonyms,
                "reference": reference,
                "ontology_id": ontology_id,
                "subclass_of": subclass_of
            })


    efo_index = build_index_efo(parsed_classes, "EFO", remove_words=False)
    efo_reduce_index = build_index_efo(parsed_classes, "EFO", remove_words=True)
    efo_fuzzy_index = build_index_efo(parsed_classes, "EFO", remove_words=False, fuzzy=True)

    with open(output_file_path, "w") as f:
        json.dump(efo_index, f, indent=2)
    with open(output_file_path_reduce, "w") as f:
        json.dump(efo_reduce_index, f, indent=2)    
    with open(output_file_path_fuzzy, "w") as f:
        json.dump(efo_fuzzy_index, f, indent=2)

def build_index_efo(data, ontology_type, remove_words=False, fuzzy=False):
    """Builds an index for EFO labels and synonyms, allowing multiple matches per term."""
    index = {}
    for item in data:
        label = item.get("label", "")
        if not isinstance(label, str) or not label.strip():
            continue

        normalized_label = process_string(label, remove=remove_words)
        if not fuzzy:
            normalized_label = remove_spaces(normalized_label)

        ontology_entry = {
            "ontology_accession": item.get("ontology_id", ""),
            "ontology_type": ontology_type,
            "official_term": label
        }

        # Add label to index
        index.setdefault(normalized_label, []).append(ontology_entry)

        # Add synonyms to index
        synonyms = item.get("exact_synonyms", []) + \
                   item.get("related_synonyms", []) + \
                   item.get("broad_synonyms", [])

        for synonym in synonyms:
            if not isinstance(synonym, str) or not synonym.strip():
                continue

            try:
                normalized_synonym = process_string(synonym, remove=remove_words)
                if not fuzzy:
                    normalized_synonym = remove_spaces(normalized_synonym)
                index.setdefault(normalized_synonym, []).append(ontology_entry)
            except Exception as e:
                continue

    return index

def process_uberon_file(input_file, output_file_path, output_file_path_reduce, output_file_path_fuzzy):
    """Parses Uberon file and extracts necessary data."""
    with open(input_file, 'r', encoding='utf-8') as infile:
        try:
            raw_data = json.load(infile)
            data = raw_data["graphs"][0]["nodes"]  # <-- this line extracts the list of entries
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error reading JSON structure: {e}")
            return

    uberon_index = build_index_uberon(data, remove_words=False, fuzzy=False)
    uberon_reduce_index = build_index_uberon(data, remove_words=True, fuzzy=False)
    uberon_fuzzy_index = build_index_uberon(data, remove_words=False, fuzzy=True)

    with open(output_file_path, "w") as f:
        json.dump(uberon_index, f, indent=2)
    with open(output_file_path_reduce, "w") as f:
        json.dump(uberon_reduce_index, f, indent=2)
    with open(output_file_path_fuzzy, "w") as f:
        json.dump(uberon_fuzzy_index, f, indent=2)

    print(f"Processed Uberon data has been saved to {output_file_path}")

def build_index_uberon(data, remove_words=False, fuzzy=False):
    """
    Builds an index for Uberon labels and exact synonyms.
    Keys are normalized strings (label and synonyms), and values are lists of matching ontology entries.
    """
    index = {}
    for item in data:
        if item is None:
            continue

        label = item.get("lbl")
        if label is None:
            continue
        ontology_id = item.get("id", "").split("/")[-1]  # extract PR_000001408, etc.

        normalized_label = process_string(label, remove=remove_words)
        if not fuzzy:
            normalized_label = remove_spaces(normalized_label)

        ontology_entry = {
            "ontology_accession": ontology_id,
            "ontology_type": "Uberon",
            "official_term": label
        }

        # Add the label
        if normalized_label in index:
            index[normalized_label].append(ontology_entry)
        else:
            index[normalized_label] = [ontology_entry]

        # Add exact synonyms
        synonyms = item.get("meta", {}).get("synonyms", [])
        for syn in synonyms:
            if syn.get("pred") == "hasExactSynonym" or syn.get("pred") == "hasBroadSynonym" or syn.get("pred") == "hasRelatedSynonym":
                synonym_text = syn.get("val", "")
                normalized_syn = process_string(synonym_text, remove=remove_words)
                if not fuzzy:
                    normalized_syn = remove_spaces(normalized_syn)
                if normalized_syn in index:
                    index[normalized_syn].append(ontology_entry)
                else:
                    index[normalized_syn] = [ontology_entry]

    return index

def process_data():
    """Processes all downloaded files, extracting relevant data and saving to parsed directories."""
    print("GEOMetaX | Processing data...")

    error_occured = False

    data_dir = get_data_dir()
    unparsed_factor_dir = data_dir / "unparsed_factor_data"
    unparsed_ontology_dir = data_dir / "unparsed_ontology_data"
    parsed_factor_dir = data_dir / "parsed_factor_data"
    parsed_ontology_dir = data_dir / "parsed_ontology_data"

    # Ensure parsed directories exist
    os.makedirs(parsed_factor_dir, exist_ok=True)
    os.makedirs(parsed_ontology_dir, exist_ok=True)

    # Process Cellosaurus
    try:
        cellosaurus_input = unparsed_ontology_dir / "cellosaurus.txt"
        cellosaurus_output = parsed_ontology_dir / "cellosaurus.json"
        cellosaurus_output_reduce = parsed_ontology_dir / "cellosaurus_reduce.json"
        cellosaurus_output_fuzzy = parsed_ontology_dir / "cellosaurus_fuzzy.json"
        if cellosaurus_input.exists():
            process_cellosaurus_file(cellosaurus_input, cellosaurus_output, cellosaurus_output_reduce, cellosaurus_output_fuzzy)
    except Exception as e:
        error_occured = True
        print(f"Error processing Cellosaurus: {e}")

    # Process EFO OWL
    try:
        efo_input = unparsed_ontology_dir / "efo.owl"
        efo_output = parsed_ontology_dir / "efo.json"
        efo_output_reduce = parsed_ontology_dir / "efo_reduce.json"
        efo_output_fuzzy = parsed_ontology_dir / "efo_fuzzy.json"
        if efo_input.exists():
            process_efo_owl(efo_input, efo_output, efo_output_reduce, efo_output_fuzzy)
    except Exception as e:
        error_occured = True
        print(f"Error processing EFO OWL: {e}")

    # Process Uberon
    try:
        uberon_input = unparsed_ontology_dir / "uberon-full.json"
        uberon_output = parsed_ontology_dir / "uberon.json"
        uberon_output_reduce = parsed_ontology_dir / "uberon_reduce.json"
        uberon_output_fuzzy = parsed_ontology_dir / "uberon_fuzzy.json"
        if uberon_input.exists():
            process_uberon_file(uberon_input, uberon_output, uberon_output_reduce, uberon_output_fuzzy)
    except Exception as e:
        error_occured = True
        print(f"Error processing Uberon: {e}")

    # Convert gene_info.gz to CSV
    try:
        # Paths
        gene_gz = unparsed_factor_dir / "gene_info.gz"
        gene_csv = parsed_factor_dir / "gene_info.csv"
        gene_ids_path = Path("GEOMetaX") / "gene_ids.npy"

        if gene_gz.exists() and gene_ids_path.exists():
            # Load valid GeneIDs into a set for fast lookup
            valid_gene_ids = set(map(str, np.load(gene_ids_path)))  # Cast to str in case of type mismatch

            with gzip.open(gene_gz, 'rt') as infile, open(gene_csv, 'w', newline='') as outfile:
                reader = csv.DictReader(infile, delimiter='\t')
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()

                match_count = 0
                for row in reader:
                    if row['GeneID'] in valid_gene_ids:
                        writer.writerow(row)
                        match_count += 1

            print(f"Saved filtered gene_info.csv with {match_count} rows to {gene_csv}")
        else:
            print("Missing gene_info.gz or gene_ids.npy")

    except Exception as e:
        error_occured = True
        print(f"Error filtering and converting gene_info.gz to CSV: {e}")

    try:
        TF_input = unparsed_factor_dir / "Homo_sapiens_TF.csv"
        TF_output = parsed_factor_dir
        if TF_input.exists():
            move_file(TF_input, TF_output)
    except Exception as e:
        error_occured = True
        print(f"Error processing Homo_sapiens_TF: {e}")

    try:
        CR_input = unparsed_factor_dir / "Homo_sapiens_CR.csv"
        CR_output = parsed_factor_dir
        if CR_input.exists():
            move_file(CR_input, CR_output)
    except Exception as e:
        error_occured = True
        print(f"Error processing Homo_sapiens_CR: {e}")

    if not error_occured:
        try:
            delete_folder(unparsed_factor_dir)
        except Exception as e:
            print(f"Error deleting Unparsed Factor Directory: {e}")
        try:
            delete_folder(unparsed_ontology_dir)
        except Exception as e:
            print(f"Error deleting Unparsed Ontology Directory: {e}")
    print("GEOMetaX | Data processing complete.")


if __name__ == "__main__":
    process_data()