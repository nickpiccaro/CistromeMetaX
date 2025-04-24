import sys
from .parser_extractor import meta_extract_one_sample

def main():
    if len(sys.argv) < 3:
        print("Usage: geoMX-extract_one_sample_file GSM_FILE GSE_FILE1 [GSE_FILE2 ...]")
        sys.exit(1)

    gsm_file = sys.argv[1]
    gse_files = sys.argv[2:]
    
    # Call your actual function
    result = meta_extract_one_sample(gsm_file, gse_files)

    print("Extraction result:")
    print(result)