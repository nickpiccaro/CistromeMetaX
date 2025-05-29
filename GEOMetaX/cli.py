import argparse
import json
import sys
from pathlib import Path
from .downloader import install_data
from .processor import process_data
from .parser_extractor import meta_extract_factors, meta_extract_ontologies, meta_extract_factors_and_ontologies


def update_data():
    if len(sys.argv) < 1:
        print("Usage: geoMX-update_data")
        sys.exit(1)
   
    install_data()
    print("Data installation complete.")
    process_data()
    print("Data processing complete.")
    print("Data update complete.")


def _parse_gsm_ids_input(gsm_ids_input):
    """
    Parse GSM IDs input - can be a list/array, JSON string, or a JSON file path.
    Returns the appropriate input format for the extraction functions.
    
    This mirrors the logic used in meta_extract_factors_and_ontologies.
    """
    # If it's already a list, return as-is (will be handled by extraction functions)
    if isinstance(gsm_ids_input, list):
        return gsm_ids_input
    
    # Handle string input
    if isinstance(gsm_ids_input, str):
        gsm_ids_input = gsm_ids_input.strip()
        
        # Try to parse as JSON string first (for direct list input)
        if gsm_ids_input.startswith('[') and gsm_ids_input.endswith(']'):
            try:
                parsed_list = json.loads(gsm_ids_input)
                if isinstance(parsed_list, list):
                    return parsed_list
            except json.JSONDecodeError:
                # Try to fix common quote issues - convert single quotes to double quotes
                try:
                    # Replace single quotes with double quotes for JSON compatibility
                    fixed_json = gsm_ids_input.replace("'", '"')
                    parsed_list = json.loads(fixed_json)
                    if isinstance(parsed_list, list):
                        return parsed_list
                except json.JSONDecodeError:
                    pass
                
                # Try to handle case where quotes are missing around strings
                try:
                    # Handle format like [GSM123, GSM456] -> ["GSM123", "GSM456"]
                    import re
                    # Find content between brackets
                    match = re.match(r'\[(.*)\]', gsm_ids_input)
                    if match:
                        content = match.group(1).strip()
                        if content:
                            # Split by comma and clean up each item
                            items = [item.strip().strip('"\'') for item in content.split(',')]
                            # Filter out empty items
                            items = [item for item in items if item]
                            if items:
                                return items
                except Exception:
                    pass
        
        # Check if it's a file path
        if Path(gsm_ids_input).exists():
            return gsm_ids_input
        else:
            # Try to parse as JSON one more time in case of formatting issues
            try:
                parsed_list = json.loads(gsm_ids_input)
                if isinstance(parsed_list, list):
                    return parsed_list
            except json.JSONDecodeError:
                pass
            
            print(f"Error: GSM IDs input '{gsm_ids_input}' is not a valid JSON list or existing file path", file=sys.stderr)
            print("Valid formats:", file=sys.stderr)
            print('  - JSON file path: gsm_ids.json', file=sys.stderr)
            print('  - JSON list: \'["GSM123", "GSM456"]\'', file=sys.stderr)
            print('  - JSON list (alt): "[\\\"GSM123\\\", \\\"GSM456\\\"]"', file=sys.stderr)
            sys.exit(1)
    
    # Handle other iterable types
    try:
        if hasattr(gsm_ids_input, 'tolist'):
            return gsm_ids_input.tolist()
        elif hasattr(gsm_ids_input, '__iter__') and not isinstance(gsm_ids_input, str):
            return list(gsm_ids_input)
        else:
            return [gsm_ids_input] if gsm_ids_input else []
    except Exception as e:
        print(f"Error parsing GSM IDs input: {e}", file=sys.stderr)
        sys.exit(1)


def meta_extract():
    """
    Console command for extracting metadata from GSM IDs.
    Supports factor extraction, ontology extraction, or both.
    """
    parser = argparse.ArgumentParser(
        description="Extract metadata (factors and/or ontologies) from GSM IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract only factors
    geoMX-extract --mode factor --gsm-ids gsm_ids.json --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json

    # Extract only ontologies
    geoMX-extract --mode ontology --gsm-ids gsm_ids.json --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json

    # Extract both factors and ontologies
    geoMX-extract --mode both --gsm-ids gsm_ids.json --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json

    # Save output to file
    geoMX-extract --mode both --gsm-ids gsm_ids.json --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json --output results.json

    # Pass GSM IDs directly as JSON list string
    geoMX-extract --mode factor --gsm-ids '["GSM123456", "GSM789012"]' --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json

    # Pass GSM IDs with different JSON formatting
    geoMX-extract --mode factor --gsm-ids "[\"GSM123456\", \"GSM789012\"]" --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json
        """
    )
   
    parser.add_argument(
        "--mode",
        choices=["factor", "ontology", "both"],
        required=True,
        help="Extraction mode: 'factor' for factors only, 'ontology' for ontologies only, 'both' for both factors and ontologies"
    )
   
    parser.add_argument(
        "--gsm-ids",
        required=True,
        help="GSM IDs input: either a path to JSON file containing GSM IDs, or a JSON string representation of a list (e.g., '[\"GSM123\", \"GSM456\"]')"
    )
   
    parser.add_argument(
        "--gsm-to-gse",
        required=True,
        help="Path to JSON file mapping GSM IDs to GSE IDs"
    )
   
    parser.add_argument(
        "--gsm-paths",
        required=True,
        help="Path to JSON file mapping GSM IDs to file paths"
    )
   
    parser.add_argument(
        "--gse-paths",
        required=True,
        help="Path to JSON file mapping GSE IDs to file paths"
    )
   
    parser.add_argument(
        "--output", "-o",
        help="Optional: Path to save the output JSON file. If not provided, results will be printed to stdout"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
   
    args = parser.parse_args()
   
    # Parse GSM IDs input using the enhanced parser
    gsm_ids_input = _parse_gsm_ids_input(args.gsm_ids)
    
    if args.verbose:
        if isinstance(gsm_ids_input, list):
            print(f"Parsed GSM IDs as list with {len(gsm_ids_input)} items")
        else:
            print(f"Using GSM IDs from file: {gsm_ids_input}")
   
    # Validate that required mapping files exist
    required_files = [args.gsm_to_gse, args.gsm_paths, args.gse_paths]
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"Error: Required mapping file not found: {file_path}", file=sys.stderr)
            sys.exit(1)
   
    # Select appropriate extraction function based on mode
    try:
        if args.mode == "factor":
            if args.verbose:
                print(f"Extracting factors for GSM IDs...")
            results = meta_extract_factors(
                gsm_ids_input,
                args.gsm_to_gse,
                args.gsm_paths,
                args.gse_paths
            )
        elif args.mode == "ontology":
            if args.verbose:
                print(f"Extracting ontologies for GSM IDs...")
            results = meta_extract_ontologies(
                gsm_ids_input,
                args.gsm_to_gse,
                args.gsm_paths,
                args.gse_paths
            )
        elif args.mode == "both":
            if args.verbose:
                print(f"Extracting both factors and ontologies for GSM IDs...")
            results = meta_extract_factors_and_ontologies(
                gsm_ids_input,
                args.gsm_to_gse,
                args.gsm_paths,
                args.gse_paths
            )
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)
   
    # Handle output
    if args.output:
        # Save to file
        output_path = Path(args.output)
        try:
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
           
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
           
            print(f"Results saved to: {output_path}")
            if isinstance(results, (list, dict)):
                if isinstance(results, list):
                    print(f"Total records processed: {len(results)}")
                elif isinstance(results, dict) and 'results' in results:
                    print(f"Total records processed: {len(results.get('results', []))}")
           
        except Exception as e:
            print(f"Error saving output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Print to stdout
        print(json.dumps(results, indent=2, ensure_ascii=False))