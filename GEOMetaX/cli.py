import sys
from .parser_extractor import meta_extract_factor, meta_extract_ontology, meta_extract_factors, meta_extract_ontologies

def one_factor():
    if len(sys.argv) < 3:
        print("Usage: geoMX-factor_extract_one GSM_FILE GSE_FILE1 [GSE_FILE2 ...]")
        sys.exit(1)

    gsm_file = sys.argv[1]
    gse_files = sys.argv[2:]
    
    # Call your actual function
    result = meta_extract_factor(gsm_file, gse_files)

    print("Extraction result:")
    print(result)

def many_factor():
    if len(sys.argv) < 2:
        print("Usage: geoMX-factor_extract_multiple JSON_FILE")
        sys.exit(1)

    json_file = sys.argv[1]
    
    # Call your actual function
    result = meta_extract_factors(json_file)

    print("Extraction result:")
    print(result)

def one_ontology():
    if len(sys.argv) < 3:
        print("Usage: geoMX-ontology_extract_one GSM_FILE GSE_FILE1 [GSE_FILE2 ...]")
        sys.exit(1)

    gsm_file = sys.argv[1]
    gse_files = sys.argv[2:]
    
    # Call your actual function
    result = meta_extract_ontology(gsm_file, gse_files)

    print("Extraction result:")
    print(result)

def many_ontology():
    if len(sys.argv) < 2:
        print("Usage: geoMX-ontology_extract_multiple JSON_FILE")
        sys.exit(1)

    json_file = sys.argv[1]
    
    # Call your actual function
    result = meta_extract_ontologies(json_file)

    print("Extraction result:")
    print(result)