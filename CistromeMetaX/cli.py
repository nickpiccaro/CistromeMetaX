import argparse
import json
import sys
from pathlib import Path
from .downloader import install_data
from .processor import process_data
from .parser_extractor import meta_extract_factors, meta_extract_ontologies, meta_extract_factors_and_ontologies


def update_data():
    if len(sys.argv) < 1:
        print("Usage: cistromeMX-update_data")
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
    Console command for extracting metadata from GSM/GSE accessions.
    Supports factor extraction, ontology extraction, or both.

    By default (no mapping JSONs supplied), MINiML XML is fetched directly
    from NCBI GEO. When all three mapping JSONs are provided, the previous
    local-file workflow is used.
    """
    parser = argparse.ArgumentParser(
        description="Extract metadata (factors and/or ontologies) from GSM/GSE accessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch mode (default) — pass bare accession strings; XML is pulled from NCBI GEO
    cistromeMX-extract --mode factor --gsm-ids '["GSM534473"]'
    cistromeMX-extract --mode both --gsm-ids '["GSM534473", "GSE20752"]' -o results.json

    # GSE accessions auto-expand to all child GSMs
    cistromeMX-extract --mode factor --gsm-ids '["GSE20752"]' -v

    # Use a different model provider in fetch mode
    cistromeMX-extract --mode both --gsm-ids '["GSM534473"]' --model anthropic:claude-sonnet-4-5-20250929

    # Local-file mode — supply all three mapping JSONs
    cistromeMX-extract --mode factor --gsm-ids gsm_ids.json --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json

    # Local-file mode with different model
    cistromeMX-extract --mode both --gsm-ids gsm_ids.json --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json --model anthropic:claude-sonnet-4-5-20250929
        """
    )

    parser.add_argument(
        "--mode",
        choices=["factor", "ontology", "both"],
        required=True,
        help="Extraction mode: 'factor', 'ontology', or 'both'"
    )

    parser.add_argument(
        "--gsm-ids",
        required=True,
        help="Accessions input: a path to a JSON file containing a list of GSM/GSE accessions, "
             "or a JSON list string (e.g., '[\"GSM534473\", \"GSE20752\"]'). "
             "GSE accessions are auto-expanded to all their child GSMs."
    )

    parser.add_argument(
        "--gsm-to-gse",
        required=False,
        default=None,
        help="Optional. Path to JSON mapping GSM IDs to GSE IDs. Required only for local-file mode; "
             "omit to fetch from NCBI."
    )

    parser.add_argument(
        "--gsm-paths",
        required=False,
        default=None,
        help="Optional. Path to JSON mapping GSM IDs to local XML file paths. Required only for "
             "local-file mode; omit to fetch from NCBI."
    )

    parser.add_argument(
        "--gse-paths",
        required=False,
        default=None,
        help="Optional. Path to JSON mapping GSE IDs to local XML file paths. Required only for "
             "local-file mode; omit to fetch from NCBI."
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

    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model to use for LLM extraction, in 'provider:model_name' format "
             "(e.g., 'openai:gpt-4o-mini', 'anthropic:claude-sonnet-4-5-20250929', "
             "'google_vertexai:gemini-2.5-flash'). Defaults to 'openai:gpt-4o-mini'. "
             "Requires the corresponding API key in your .env file and the provider's "
             "langchain integration package installed (e.g., 'pip install langchain-anthropic')."
    )
   
    args = parser.parse_args()

    # Parse GSM IDs input using the enhanced parser
    gsm_ids_input = _parse_gsm_ids_input(args.gsm_ids)

    # Decide between fetch mode and local-file mode based on mapping args.
    # All three mapping JSONs absent => fetch mode. All three present => local mode.
    # Anything in between is a usage error.
    mapping_args = (args.gsm_to_gse, args.gsm_paths, args.gse_paths)
    fetch_mode = all(p is None for p in mapping_args)
    if not fetch_mode and not all(mapping_args):
        print(
            "Error: provide all three mapping files (--gsm-to-gse, --gsm-paths, --gse-paths) "
            "for local-file mode, or omit all three to fetch from NCBI.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.verbose:
        if isinstance(gsm_ids_input, list):
            print(f"Parsed GSM IDs as list with {len(gsm_ids_input)} items")
        else:
            print(f"Using GSM IDs from file: {gsm_ids_input}")
        if args.model:
            print(f"Using model: {args.model}")
        else:
            print(f"Using default model: openai:gpt-4o-mini")
        print(f"Mode: {'fetch from NCBI GEO' if fetch_mode else 'local files'}")

    # Validate mapping files only when in local-file mode
    if not fetch_mode:
        for file_path in mapping_args:
            if not Path(file_path).exists():
                print(f"Error: Required mapping file not found: {file_path}", file=sys.stderr)
                sys.exit(1)

    # Build kwargs that work for both modes
    extract_kwargs = dict(model=args.model, verbose=args.verbose)
    if not fetch_mode:
        extract_kwargs.update(
            gsm_to_gse_path=args.gsm_to_gse,
            gsm_paths_path=args.gsm_paths,
            gse_paths_path=args.gse_paths,
        )

    # Select appropriate extraction function based on mode
    try:
        if args.mode == "factor":
            if args.verbose:
                print(f"Extracting factors for GSM IDs...")
            results = meta_extract_factors(gsm_ids_input, **extract_kwargs)
        elif args.mode == "ontology":
            if args.verbose:
                print(f"Extracting ontologies for GSM IDs...")
            results = meta_extract_ontologies(gsm_ids_input, **extract_kwargs)
        elif args.mode == "both":
            if args.verbose:
                print(f"Extracting both factors and ontologies for GSM IDs...")
            results = meta_extract_factors_and_ontologies(gsm_ids_input, **extract_kwargs)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
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