# CistromeMetaX

A Python package and command-line tool that leverages large language models (LLMs) to parse, extract, and verify metadata from GEO MetaData XML files (from the NCBI Gene Expression Omnibus) specifically for ChIP-seq experiments. CistromeMetaX extracts crucial experimental factors, cell types, tissues, and target proteins in formats useful for downstream tools such as the [Cistrome Data Browser](https://db3.cistrome.org/browser/).

## The Challenge

Manual metadata extraction for ChIP-seq experiments is an extraordinarily time-consuming and often impractical approach. As ChIP-seq experiments become increasingly affordable and accessible, the volume of open-access datasets continues to grow exponentially. Unfortunately, the variety of metadata practices across different laboratories and research groups means that many powerful ChIP-seq datasets remain underutilized because their metadata cannot be efficiently standardized for computational analysis.

This presents a significant bottleneck: thousands of valuable ChIP-seq experiments sit in public repositories with inconsistent or incomplete metadata annotations, preventing researchers from leveraging these datasets for meta-analyses, comparative studies, and broader biological insights. CistromeMetaX addresses this critical gap by utilizing advanced language models to automatically extract and verify key metadata values, enabling the standardization and integration of diverse ChIP-seq datasets for powerful data-driven discoveries.

---

## Table of Contents

- [About](#about)
- [The Model](#the-model)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python Interface](#python-interface)
- [Input File Structure](#input-file-structure)
- [Expected Output](#expected-output)
- [Generating Input Files](#generating-input-files)
- [Changelog](#changelog)
- [Future Goals](#future-goals)
- [References](#references)
- [Support](#support)

---

## About

CistromeMetaX streamlines the extraction of critical metadata from ChIP-seq experiments, including experimental factors, cell types, tissues, and target proteins from GEO (Gene Expression Omnibus) XML files. Unlike other metadata extraction tools, CistromeMetaX is built on native ChatGPT OpenAI functionality and requires no model training or specialized setup. The package validates its LLM outputs against established databases to ensure extracted cell types, tissues, cell lines, and target proteins are biologically valid and standardized.

The tool is designed to integrate seamlessly with existing bioinformatics pipelines, providing highly accurate and consistent outputs suitable for resources like Cistrome and other ChIP-seq analysis platforms.

---

## The Model

This package utilizes OpenAI's GPT models through their native API. It performs semantic parsing and context-aware extraction to generate structured metadata from ChIP-seq experiment descriptions. The results are validated against established biological databases and optimized for minimal post-processing, allowing direct integration into downstream databases or analytical tools.

---

## Requirements

- Python 3.6+
- OpenAI API Key
- Virtual environment (recommended)

---

## Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/nickpiccaro/CistromeMetaX.git
```

### Setup Instructions

1. **Create a virtual environment** using Python 3:

    ```bash
    python3 -m venv envCistromeMetaX
    ```

2. **Activate the virtual environment**:

    - **Bash**:

      ```bash
      source envCistromeMetaX/Scripts/activate
      ```

    - **PowerShell**:

      ```powershell
      .\envCistromeMetaX\Scripts\Activate
      ```

3. **Get your OpenAI API Key**:
   - Visit [OpenAI's API platform](https://platform.openai.com/api-keys)
   - Sign up or log in to your account
   - Navigate to "API Keys" in your dashboard
   - Click "Create new secret key"
   - Copy the generated key (it will only be shown once)

4. **Add your OpenAI API Key** to a `.env` file in your project directory:

    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

---

## Usage

### Command Line Interface

The new streamlined CLI command uses JSON configuration files for batch processing:

```bash
geoMX-extract --mode [factor|ontology|both] --gsm-ids GSM_IDS_INPUT --gsm-to-gse GSM_TO_GSE_FILE --gsm-paths GSM_PATHS_FILE --gse-paths GSE_PATHS_FILE [--output OUTPUT_FILE] [--verbose]
```

#### CLI Arguments

- `--mode`: Extraction mode
  - `factor`: Extract experimental factors only
  - `ontology`: Extract cell types and tissues only  
  - `both`: Extract both factors and cell types/tissues
- `--gsm-ids`: GSM IDs input (JSON file path or JSON string)
- `--gsm-to-gse`: Path to JSON file mapping GSM IDs to GSE IDs
- `--gsm-paths`: Path to JSON file mapping GSM IDs to file paths
- `--gse-paths`: Path to JSON file mapping GSE IDs to file paths
- `--output, -o`: Optional output file path (prints to stdout if not specified)
- `--verbose, -v`: Enable verbose output

#### Example Usage

```bash
# Extract only factors
geoMX-extract --mode factor --gsm-ids gsm_ids.json --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json

# Extract cell types and tissues, save to file
geoMX-extract --mode ontology --gsm-ids gsm_ids.json --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json --output results/cell_types.json

# Extract both factors and cell types/tissues
geoMX-extract --mode both --gsm-ids gsm_ids.json --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json -o combined_results.json

# Pass GSM IDs directly as JSON string
geoMX-extract --mode factor --gsm-ids '["GSM123456", "GSM789012"]' --gsm-to-gse mappings/gsm_to_gse.json --gsm-paths mappings/gsm_paths.json --gse-paths mappings/gse_paths.json
```

---

### Python Interface

You can use CistromeMetaX directly within your Python scripts:

```python
from CistromeMetaX import meta_extract_factors, meta_extract_ontologies, meta_extract_factors_and_ontologies
import json

# Example 1: Extract both factors and cell types/tissues using file paths
result = meta_extract_factors_and_ontologies(
    gsm_ids_input="metadata/gsm_ids.json",
    gsm_to_gse_path="metadata/gsm_to_gse.json", 
    gsm_paths_path="metadata/gsm_paths.json",
    gse_paths_path="metadata/gse_paths.json"
)

with open("full_results.json", 'w') as f:
    json.dump(result, f, indent=4)
print(result)

# Example 2: Extract factors only using direct GSM ID list
gsm_ids_input = ["GSM669931", "GSM1006151"]
result_factors = meta_extract_factors(
    gsm_ids_input=gsm_ids_input,
    gsm_to_gse_path="metadata/gsm_to_gse.json", 
    gsm_paths_path="metadata/gsm_paths.json",
    gse_paths_path="metadata/gse_paths.json"
)

# Example 3: Extract cell types and tissues only
result_ontologies = meta_extract_ontologies(
    gsm_ids_input="metadata/gsm_ids.json",
    gsm_to_gse_path="metadata/gsm_to_gse.json", 
    gsm_paths_path="metadata/gsm_paths.json",
    gse_paths_path="metadata/gse_paths.json"
)
```

---

## Input File Structure

CistromeMetaX requires four JSON configuration files to process your ChIP-seq metadata:

### 1. GSM IDs File (`gsm_ids.json`)
List of GSM identifiers to process:
```json
[
  "GSM1006151",
  "GSM1007988",
  "GSM1009641",
  "GSM1013129"
]
```

### 2. GSM to GSE Mapping (`gsm_to_gse.json`)
Maps each GSM to its associated GSE experiments:
```json
{
  "GSM1006151": [
    "GSE40970",
    "GSE40972"
  ],
  "GSM1007988": [
    "GSE41048",
    "GSE41050"
  ],
  "GSM1009641": [
    "GSE41166"
  ]
}
```

### 3. GSM File Paths (`gsm_paths.json`)
Maps GSM IDs to their XML file locations:
```json
{
  "GSM1006151": "path/to/GSM1006151.xml",
  "GSM1007988": "path/to/GSM1007988.xml",
  "GSM1009641": "path/to/GSM1009641.xml"
}
```

### 4. GSE File Paths (`gse_paths.json`)
Maps GSE IDs to their XML file locations:
```json
{
  "GSE40970": "path/to/GSE40970.xml",
  "GSE40972": "path/to/GSE40972.xml",
  "GSE41048": "path/to/GSE41048.xml",
  "GSE41050": "path/to/GSE41050.xml"
}
```

**Example GEO XML File**: You can view an example of what a GEO XML file looks like [here](https://github.com/nickpiccaro/CistromeMetaX/blob/main/sample.xml).

---

## Expected Output

CistromeMetaX produces structured JSON output containing extracted and validated metadata:

### Factor Extraction Output
# SCRUB FIX
```json
{
  "GSM1007988": {
      "factor": {
          "extracted_factor": "H3K27me3"
      }
  }
}
```

### Cell Type/Tissue Extraction Output  
```json
{
  "GSM1007988": {
      "ontology": {
          "extracted_ontologies": {
              "cell_line": [
                  {
                      "official_term": "WI38",
                      "term_identity": "cell_line",
                      "ontology_accession": "EFO_0001260",
                      "term": "WI-38",
                      "ontology_type": "EFO"
                  }
              ],
              "cell_type": [
                  {
                      "official_term": "fibroblast",
                      "term_identity": "cell_type",
                      "ontology_accession": "CL_0000057",
                      "term": "fibroblast",
                      "ontology_type": [
                          "EFO",
                          "Uberon"
                      ]
                  }
              ],
              "tissue": [
                  {
                      "official_term": "lung neoplasm",
                      "term_identity": "tissue",
                      "ontology_accession": "MONDO_0021117",
                      "term": "lung",
                      "ontology_type": "EFO"
                  },
                  {
                      "official_term": "lung",
                      "term_identity": "tissue",
                      "ontology_accession": "UBERON_0002048",
                      "term": "lung",
                      "ontology_type": [
                          "EFO",
                          "Uberon"
                      ]
                  }
              ],
              "disease": "N/A"
          }
      }
  }
}
```

### Combined Output (Both Mode)
```json
{
  "GSM1007988": {
      "factor": {
          "extracted_factor": "H3K27me3"
      },
      "ontology": {
          "extracted_ontologies": {
              "cell_line": [
                  {
                      "official_term": "WI38",
                      "term_identity": "cell_line",
                      "ontology_accession": "EFO_0001260",
                      "term": "WI-38",
                      "ontology_type": "EFO"
                  }
              ],
              "cell_type": [
                  {
                      "official_term": "fibroblast",
                      "term_identity": "cell_type",
                      "ontology_accession": "CL_0000057",
                      "term": "fibroblast",
                      "ontology_type": [
                          "EFO",
                          "Uberon"
                      ]
                  }
              ],
              "tissue": [
                  {
                      "official_term": "lung neoplasm",
                      "term_identity": "tissue",
                      "ontology_accession": "MONDO_0021117",
                      "term": "lung",
                      "ontology_type": "EFO"
                  },
                  {
                      "official_term": "lung",
                      "term_identity": "tissue",
                      "ontology_accession": "UBERON_0002048",
                      "term": "lung",
                      "ontology_type": [
                          "EFO",
                          "Uberon"
                      ]
                  }
              ],
              "disease": "N/A"
          }
      }
  }
}

```

---

## Generating Input Files

If you need to create the required JSON input files from your existing data structure, use this AI prompt to generate a custom Python function:

### AI Prompt Template

```
I need to create JSON configuration files for CistromeMetaX from my existing ChIP-seq data organization. 

**My current data structure:**
[Describe how your GSM and GSE XML files are currently organized, including directory structure and naming conventions]

**Required output files:**
1. gsm_ids.json - Array of GSM identifiers: ["GSM123", "GSM456", ...]
2. gsm_to_gse.json - Object mapping GSM to GSE arrays: {"GSM123": ["GSE789"], ...}  
3. gsm_paths.json - Object mapping GSM IDs to XML file paths: {"GSM123": "path/to/GSM123.xml", ...}
4. gse_paths.json - Object mapping GSE IDs to XML file paths: {"GSE789": "path/to/GSE789.xml", ...}

Please generate a Python function that reads my data structure and creates these four JSON files with the correct format for CistromeMetaX.
```

---

## Changelog

### Added

- (5/29/25) New streamlined CLI interface with JSON configuration files
- (5/29/25) Support for direct GSM ID list input in Python interface
- (5/29/25) Enhanced validation against biological databases
- (5/29/25) ChIP-seq specific metadata extraction optimizations
- (5/22/25) Initial CLI and Python interface for cell type/tissue extraction
- (5/22/25) Support for batch JSON-based parsing

### Changed

- (5/29/25) Renamed package from GEOMetaX to CistromeMetaX
- (5/29/25) Updated terminology from "ontology" to "cell types and tissues"
- (5/29/25) Restructured CLI to use JSON configuration files

### Removed

- (5/29/25) Legacy CLI commands (replaced with unified `geoMX-extract`)

---

## Future Goals

- Expand LLM support to other providers (e.g., Claude, Mistral)
- Support async batch processing for large-scale datasets
- Real-time metadata quality assessment
- Extract additional features (e.g., chemical and experimental modifications)

---

## References

- [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/)
- [Cistrome Data Browser](https://db3.cistrome.org/browser/)
- [OpenAI GPT Models](https://platform.openai.com/docs)
- [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/) - Gene/Target Protein Validation
- [Harmonize 3.0](https://maayanlab.cloud/Harmonizome) - Chromatin Remodelers Validation Data
- [AnimalTFDB v4.0](https://guolab.wchscu.cn/AnimalTFDB4//#/) - Animal Transcription Factor Database
- [Cellosaurus](https://www.cellosaurus.org/) - Cell Line Database
- [EFO](https://github.com/EBISPOT/efo/?tab=readme-ov-file) - Experimental Factor Ontology Database
- [Uberon](https://obophenotype.github.io/uberon/) - Anatomical Ontology Database

---

## Support

For issues, questions, or feature requests, please reach out via email at npiccaro [dot] business [at] gmail [dot] com.

---