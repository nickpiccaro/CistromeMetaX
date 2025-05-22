import sys
from .parser_extractor import meta_extract_factor, meta_extract_ontology

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