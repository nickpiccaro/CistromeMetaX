# GEOMetaX

A Python package and command-line tool that leverages large language models (LLMs) to parse, extract, and verify metadata from GEO MetaData XML files (from the NCBI Gene Expression Omnibus). It extracts crucial experimental factor and ontology information in formats useful for downstream tools such as the [Cistrome Data Browser](https://db3.cistrome.org/browser/). GEOMetaX aims to reduce the need for time-consuming and expensive manual metadata curation.

SCRUB ChipSeq usage, describe outputs, use cell types and tissues instead, mention that this uses chatgpt doesnt require training, and validates on 
How to get an API key for openAI, really improve the input
---
# NAME IDEAS
- CistromeGEOX
- CistromeParser
- GEOMetaX
- CistromeGEOlator
- GEOParseAI
- GEO2Cistrome
- CistromeMeta


## Table of Contents

- [About](#about)
- [The Model](#the-model)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python Interface](#python-interface)
- [Changelog](#changelog)
  - [Removed](#removed)
  - [Changed](#changed)
  - [Added](#added)
- [Future Goals](#future-goals)
- [References](#references)
- [Support](#support)

---

## About

GEOMetaX streamlines the extraction of critical metadata such as experimental factors and ontologies from GEO (Gene Expression Omnibus) XML files using LLMs. It is designed to integrate with existing bioinformatics pipelines, providing highly accurate and consistent outputs suitable for resources like Cistrome.

---

## The Model

This package utilizes the OpenAI GPT models under the hood. It performs semantic parsing and context-aware extraction to generate structured metadata. The results are optimized for minimal post-processing and can be fed directly into downstream databases or tools.

---

## Requirements

- Python 3.6+
- OpenAI API Key
- Virtual environment (recommended)

---

## Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/nickpiccaro/GEOMetaX.git
```

### Setup Instructions

1. **Create a virtual environment** using Python 3:

    ```bash
    python3 -m venv envGEOMetaX
    ```

2. **Activate the virtual environment**:

    - **Bash**:

      ```bash
      source envGEOMetaX/Scripts/activate
      ```

    - **PowerShell**:

      ```powershell
      .\envGEOMetaX\Scripts\Activate
      ```

3. **Add your OpenAI API Key** to a `.env` file in your project directory:

    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

---

## Usage

### Command Line Interface

You can use the following CLI commands after installation:
https://docs.python.org/3/library/argparse.html
-add a save flag
```bash
geoMX-update_data
geoMX-factor_extract_one GSM_FILE GSE_FILE1 [GSE_FILE2 ...]
geoMX-ontology_extract_one GSM_FILE GSE_FILE1 [GSE_FILE2 ...]
geoMX-factor_extract_multiple JSON_FILE
geoMX-ontology_extract_multiple JSON_FILE
geoMX-extract_all JSON_FILE
```

#### CLI Command Descriptions

- `geoMX-update_data`  
  Updates internal ontology and factor data. Not required unless upstream sources are updated.

- `geoMX-factor_extract_one GSM_FILE GSE_FILE1 [GSE_FILE2 ...]`  
  Extracts factors for a single GSM file and its associated GSE files.

- `geoMX-ontology_extract_one GSM_FILE GSE_FILE1 [GSE_FILE2 ...]`  
  Extracts ontology terms for a single GSM file and its associated GSE files.

- `geoMX-factor_extract_multiple JSON_FILE`  
  Extracts factors from multiple files listed in a JSON object.

- `geoMX-ontology_extract_multiple JSON_FILE`  
  Extracts ontology terms from multiple files listed in a JSON object.

- `geoMX-extract_all JSON_FILE`  
  Extracts both factors and ontology terms from multiple files via JSON.

#### Example Usage
```bash
geoMX-factor_extract_one "path/to/GSM353611.xml" "path/to/GSE14097.xml" "path/to/GSE14092.xml"
```

```bash
geoMX-ontology_extract_multiple "path/to/many_GSMs.json"
```
##### JSON Input Format for CLI (for `*_multiple` or `extract_all` commands)
```json
[
    {
        "gsm_file_path": "path/to/GSM353611.xml",
        "gse_file_paths": [
            "path/to/GSE14097.xml",
            "path/to/GSE14092.xml"
        ]
    },
    {
        "gsm_file_path": "path/to/GSM353617.xml",
        "gse_file_paths": []
    },
    {
        "gsm_file_path": "path/to/GSM448027.xml",
        "gse_file_paths": [
            "path/to/GSE17937.xml"
        ]
    }
]
```

---

### Python Interface

You can also use GEOMetaX directly within your Python scripts:

```python
from GEOMetaX import meta_extract_ontology, meta_extract_factor, meta_extract_factors, meta_extract_ontologies, meta_extract_factors_and_ontologies
import json

# Single extraction example
gsm_file_path = 'GSM669931/GSM/GSM669931.xml'
gse_file_paths = ["GSM669931/GSE/GSE14097.xml"]

result1 = meta_extract_factor(gsm_file_path, gse_file_paths)
with open("test.json", 'w') as f:
    json.dump(result1, f, indent=4)
print(result1)

# Batch extraction example using JSON input
json_file_path = 'json_test.json'
result = meta_extract_ontologies(json_file_path)
with open("test_output.json", 'w') as f:
    json.dump(result, f, indent=4)
print(result)
```

---

## Changelog

### Removed

- _To be added in future versions_

### Changed

- _To be added in future versions_

### Added

- (5/22/25) Initial CLI and Python interface for ontology/factor extraction
- (5/22/25) Support for batch JSON-based parsing

---

## Future Goals

- Expand LLM support to other providers (e.g., Claude, Mistral)
- Support async batch processing
- General validation/debugging

---

## References

- [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/)
- [Cistrome Data Browser](https://db3.cistrome.org/browser/)
- [OpenAI GPT Models](https://platform.openai.com/docs)
- [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/) Gene/Target Protein Validation
- [Harmonize 3.0](https://maayanlab.cloud/Harmonizome) Chromatin Remodelers Validation Data
- [AnimalTFDB v4.0](https://guolab.wchscu.cn/AnimalTFDB4//#/) Animal Transcription Factor Database
- [Cellosaurus](https://www.cellosaurus.org/) Cell line Database
- [EFO](https://github.com/EBISPOT/efo/?tab=readme-ov-file) Ontology Database
- [Uberon](https://obophenotype.github.io/uberon/) Ontology Database
---

## Support

For issues, please reach out via email at nickpiccaro [at] gmail [dot] com.

---
```