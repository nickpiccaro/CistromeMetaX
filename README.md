# CistromeMetaX

A Python package and command-line tool that leverages large language models (LLMs) to parse, extract, and verify metadata from GEO MetaData XML files (from the NCBI Gene Expression Omnibus) specifically for ChIP-seq experiments. **Give it a list of GSM (or GSE) accession IDs and it pulls the MINiML XML directly from NCBI GEO and returns standardized factors, cell types, tissues, and target proteins** — formats useful for downstream tools such as the [Cistrome Data Browser](https://db3.cistrome.org/browser/). For users who already have local XML files cached, the original local-file workflow is still supported.

## The Challenge

Manual metadata extraction for ChIP-seq experiments is an extraordinarily time-consuming and often impractical approach. As ChIP-seq experiments become increasingly affordable and accessible, the volume of open-access datasets continues to grow exponentially. Unfortunately, the variety of metadata practices across different laboratories and research groups means that many powerful ChIP-seq datasets remain underutilized because their metadata cannot be efficiently standardized for computational analysis.

This presents a significant bottleneck: thousands of valuable ChIP-seq experiments sit in public repositories with inconsistent or incomplete metadata annotations, preventing researchers from leveraging these datasets for meta-analyses, comparative studies, and broader biological insights. CistromeMetaX addresses this critical gap by utilizing advanced language models to automatically extract and verify key metadata values, enabling the standardization and integration of diverse ChIP-seq datasets for powerful data-driven discoveries.

---

## Table of Contents

- [About](#about)
- [Requirements](#requirements)
- [Installation](#installation)
- [The Model](#the-model)
- [Model Configuration](#model-configuration)
- [Usage](#usage)
  - [Quick Start: Accession IDs (Fetch from GEO)](#quick-start-accession-ids-fetch-from-geo)
  - [Local-File Mode](#local-file-mode)
- [Input File Structure (Local-File Mode Only)](#input-file-structure-local-file-mode-only)
- [Expected Output](#expected-output)
- [Factor Classification Categories](#factor-classification-categories)
- [Generating Input Files](#generating-input-files)
- [Changelog](#changelog)
- [Future Goals](#future-goals)
- [References](#references)
- [Support](#support)

---

## About

CistromeMetaX streamlines the extraction of critical metadata from ChIP-seq experiments, including experimental factors, cell types, tissues, and target proteins from GEO (Gene Expression Omnibus) records. The default workflow takes a list of GSM/GSE accession strings and fetches the MINiML XML directly from NCBI GEO — no pre-downloaded files or mapping JSONs required. For pipelines with locally cached XMLs, the package also accepts pre-built mapping files. CistromeMetaX supports multiple LLM providers out of the box and validates its LLM outputs against established databases to ensure extracted cell types, tissues, cell lines, and target proteins are biologically valid and standardized.

The tool is designed to integrate seamlessly with existing bioinformatics pipelines, providing highly accurate and consistent outputs suitable for resources like Cistrome and other ChIP-seq analysis platforms.

---

## Requirements

- Python 3.6+
- An API key for at least one supported LLM provider
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

3. **Install CistromeMetaX**:

    ```bash
    pip install git+https://github.com/nickpiccaro/CistromeMetaX.git
    ```

4. **Add your API key(s)** to a `.env` file in your project directory (see [Model Configuration](#model-configuration) below):

    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

---

## The Model

CistromeMetaX uses [LangChain's `init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/) to provide a unified interface across LLM providers. It performs semantic parsing and context-aware extraction to generate structured metadata from ChIP-seq experiment descriptions. The results are validated against established biological databases and optimized for minimal post-processing, allowing direct integration into downstream databases or analytical tools.

By default, CistromeMetaX uses OpenAI's `gpt-4o-mini`, but you can use any supported LLM provider by specifying the `--model` flag on the CLI or passing the `model` parameter in the Python API.

### Prompt Caching

The large guideline prompts that drive factor and ontology extraction are structured as byte-identical static prefixes so they can be served from each provider's prompt cache whenever possible. This happens automatically — no configuration required:

- **OpenAI**, **Google Gemini**, and **DeepSeek** apply prompt caching server-side when a sufficiently long static prefix is reused.
- **Anthropic** receives an explicit `cache_control` marker on the static guidelines so cache reads can be billed at the discounted rate.
- **Mistral** and any other provider fall back to a plain system message with no cache marker — output is unchanged, just no caching discount.

In addition, the factor extractor and its fallback recheck step share the **same** static prefix, so a fallback invocation is a cache hit rather than a fresh cache write. To see cache-usage telemetry from your LLM provider on stderr during a run, set `CISTROMEMX_CACHE_DEBUG=1` in your environment.

> Note: Google Gemini's implicit caching is server-side best-effort — a `cache_read` of zero in telemetry doesn't indicate a problem with CistromeMetaX and may simply reflect Google's caching heuristics on a given run.

---

## Model Configuration

### Supported Providers (Pre-installed)

CistromeMetaX ships with LangChain integration packages for the following providers — no additional installation required:

| Provider | Model String Example | Required Env Variable | Get an API Key |
|----------|---------------------|-----------------------|----------------|
| **OpenAI** (default) | `openai:gpt-4o-mini` | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| **Anthropic** | `anthropic:claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| **Google GenAI** | `google_genai:gemini-2.5-flash` | `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| **Mistral AI** | `mistralai:mistral-large-latest` | `MISTRAL_API_KEY` | [console.mistral.ai](https://console.mistral.ai/) |
| **DeepSeek** | `deepseek:deepseek-chat` | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com/) |

### Using Additional Providers

CistromeMetaX supports any LLM provider that has a LangChain chat model integration package. To use a provider not listed above, simply install its package into your environment:

```bash
pip install langchain-<provider>
```

Then pass its model string to the `--model` flag or `model` parameter as usual. For a full list of supported providers and their model strings, see the [LangChain Chat Model Integrations](https://docs.langchain.com/oss/python/integrations/chat) page.

### Model String Format

Models are specified in `provider:model_name` format. If no model is specified, CistromeMetaX defaults to `openai:gpt-4o-mini`.

```
openai:gpt-4o-mini                       # OpenAI GPT-4o Mini (default)
openai:gpt-4o                            # OpenAI GPT-4o
anthropic:claude-sonnet-4-5-20250929     # Anthropic Claude Sonnet
google_genai:gemini-2.5-flash            # Google Gemini Flash
mistralai:mistral-large-latest            # Mistral Large
deepseek:deepseek-chat                    # DeepSeek Chat
```

### Setting Up Your API Keys

Create a `.env` file in your project directory with the API key(s) for the provider(s) you plan to use. You only need to set the key for the provider you're actively using:

```env
# Required for default usage (OpenAI)
OPENAI_API_KEY=sk-proj-your-key-here

# Optional — only needed if you use these providers
ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY=your-google-key-here
MISTRAL_API_KEY=your-mistral-key-here
DEEPSEEK_API_KEY=your-deepseek-key-here
```

---

## Usage

CistromeMetaX has two operating modes:

- **Fetch mode (default)** — pass a list of GSM/GSE accession strings and let CistromeMetaX pull the MINiML XML directly from NCBI GEO. No mapping files required.
- **Local-file mode** — pass three pre-built JSON mapping files plus a list of GSM IDs. Use this when you have GEO XMLs already cached on disk and want to avoid network calls.

### Quick Start: Accession IDs (Fetch from GEO)

#### Command Line

```bash
# Extract factors for one GSM — fetches XML from NCBI in memory
cistromeMX-extract --mode factor --gsm-ids '["GSM534473"]'

# Extract both factors and ontologies, save to file, verbose progress
cistromeMX-extract --mode both --gsm-ids '["GSM534473", "GSM669931"]' -o results.json -v

# Pass a GSE — auto-expands to all child GSMs
cistromeMX-extract --mode factor --gsm-ids '["GSE20752"]' -v

# Use a different model provider
cistromeMX-extract --mode both --gsm-ids '["GSM534473"]' \
  --model anthropic:claude-sonnet-4-5-20250929
```

In fetch mode, CistromeMetaX:

- Hits the public NCBI MINiML endpoint (`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi`) with rate limiting (≥0.35s between requests) and up to 3 retries per accession.
- Auto-discovers each GSM's parent GSE(s) using `targ=series`, falling back to `targ=all`, so the LLM gets the same GSM + GSE context as in local-file mode.
- Expands any GSE accession in the input list to all of its child GSMs (de-duplicated against any GSMs you also listed).
- Holds fetched XMLs in memory only — nothing is written to disk.
- After 3 failed retries on an accession, prints a `Could not fetch <accession>` line to stderr and emits `{<accession>: {}}` in the output list, preserving output positions for downstream consumers.

#### Python

```python
from CistromeMetaX import (
    meta_extract_factors,
    meta_extract_ontologies,
    meta_extract_factors_and_ontologies,
)

# Default: fetch from NCBI. Pass GSM and/or GSE accessions.
result = meta_extract_factors_and_ontologies(["GSM534473", "GSE20752"])

# Verbose progress on stderr; pick a specific model
result = meta_extract_factors(
    ["GSM534473"],
    model="anthropic:claude-sonnet-4-5-20250929",
    verbose=True,
)
```

You can also call the fetch helpers directly:

```python
from CistromeMetaX import fetch_geo_xml, expand_gse_to_gsms

xml_str = fetch_geo_xml("GSM534473", targ="series")  # MINiML XML as a string
child_gsms = expand_gse_to_gsms("GSE20752")           # list of child GSM accessions
```

---

### Local-File Mode

Use this mode when you already have GEO XML files cached locally and pre-built mapping JSONs. Behavior here is unchanged from previous releases.

#### Command Line

```bash
cistromeMX-extract --mode [factor|ontology|both] \
  --gsm-ids GSM_IDS_INPUT \
  --gsm-to-gse GSM_TO_GSE_FILE \
  --gsm-paths GSM_PATHS_FILE \
  --gse-paths GSE_PATHS_FILE \
  [--model MODEL] [--output OUTPUT_FILE] [--verbose]
```

#### CLI Arguments

- `--mode`: Extraction mode
  - `factor`: Extract experimental factors only
  - `ontology`: Extract cell types and tissues only
  - `both`: Extract both factors and cell types/tissues
- `--gsm-ids`: GSM/GSE accessions input (JSON file path or JSON string). In fetch mode, GSE accessions auto-expand to child GSMs.
- `--gsm-to-gse`: *(Optional — local-file mode only)* Path to JSON file mapping GSM IDs to GSE IDs.
- `--gsm-paths`: *(Optional — local-file mode only)* Path to JSON file mapping GSM IDs to file paths.
- `--gse-paths`: *(Optional — local-file mode only)* Path to JSON file mapping GSE IDs to file paths.
- `--model, -m`: LLM model in `provider:model_name` format (default: `openai:gpt-4o-mini`).
- `--output, -o`: Optional output file path (prints to stdout if not specified).
- `--verbose, -v`: Enable verbose output (stderr).

> **Note:** All three mapping flags must be provided together for local-file mode. Omit all three to use fetch mode. Providing some-but-not-all is an error.

#### Example Usage

```bash
# Extract both factors and ontologies (default model: openai:gpt-4o-mini)
cistromeMX-extract --mode both \
  --gsm-ids gsm_ids.json \
  --gsm-to-gse mappings/gsm_to_gse.json \
  --gsm-paths mappings/gsm_paths.json \
  --gse-paths mappings/gse_paths.json \
  -o results.json

# Use Anthropic Claude
cistromeMX-extract --mode both \
  --gsm-ids gsm_ids.json \
  --gsm-to-gse mappings/gsm_to_gse.json \
  --gsm-paths mappings/gsm_paths.json \
  --gse-paths mappings/gse_paths.json \
  --model anthropic:claude-sonnet-4-5-20250929

# Use Google Gemini for factor extraction only
cistromeMX-extract --mode factor \
  --gsm-ids gsm_ids.json \
  --gsm-to-gse mappings/gsm_to_gse.json \
  --gsm-paths mappings/gsm_paths.json \
  --gse-paths mappings/gse_paths.json \
  -m google_genai:gemini-2.5-flash

# Pass GSM IDs directly as JSON string (still local-file mode here, since mappings are supplied)
cistromeMX-extract --mode factor \
  --gsm-ids '["GSM123456", "GSM789012"]' \
  --gsm-to-gse mappings/gsm_to_gse.json \
  --gsm-paths mappings/gsm_paths.json \
  --gse-paths mappings/gse_paths.json
```

#### Python

```python
from CistromeMetaX import meta_extract_factors, meta_extract_ontologies, meta_extract_factors_and_ontologies
import json

# Example 1: Extract both factors and cell types/tissues (default model)
result = meta_extract_factors_and_ontologies(
    gsm_ids_input="metadata/gsm_ids.json",
    gsm_to_gse_path="metadata/gsm_to_gse.json",
    gsm_paths_path="metadata/gsm_paths.json",
    gse_paths_path="metadata/gse_paths.json"
)

with open("full_results.json", 'w') as f:
    json.dump(result, f, indent=4)

# Example 2: Use Anthropic Claude
result = meta_extract_factors_and_ontologies(
    gsm_ids_input="metadata/gsm_ids.json",
    gsm_to_gse_path="metadata/gsm_to_gse.json",
    gsm_paths_path="metadata/gsm_paths.json",
    gse_paths_path="metadata/gse_paths.json",
    model="anthropic:claude-sonnet-4-5-20250929"
)

# Example 3: Extract factors only with Google Gemini
gsm_ids_input = ["GSM669931", "GSM1006151"]
result_factors = meta_extract_factors(
    gsm_ids_input=gsm_ids_input,
    gsm_to_gse_path="metadata/gsm_to_gse.json",
    gsm_paths_path="metadata/gsm_paths.json",
    gse_paths_path="metadata/gse_paths.json",
    model="google_genai:gemini-2.5-flash"
)

# Example 4: Extract cell types and tissues only (default model)
result_ontologies = meta_extract_ontologies(
    gsm_ids_input="metadata/gsm_ids.json",
    gsm_to_gse_path="metadata/gsm_to_gse.json",
    gsm_paths_path="metadata/gsm_paths.json",
    gse_paths_path="metadata/gse_paths.json"
)
```

---

## Input File Structure (Local-File Mode Only)

The four JSON configuration files below are only required for **local-file mode**. Fetch mode (the default) does not need any of them — pass accession strings to `--gsm-ids` and CistromeMetaX will discover everything else from NCBI.

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

Every successful factor extraction includes both `extracted_factor` and `factor_type`:

```json
{
  "GSM1007988": {
      "factor": {
          "extracted_factor": "H3K27me3",
          "factor_type": "histone_modification"
      }
  }
}
```

A transcription-factor example:

```json
{
  "GSM534473": {
      "factor": {
          "extracted_factor": "ESR1",
          "factor_type": "transcription_factor"
      }
  }
}
```

A viral-factor example (KSHV-infected cells):

```json
{
  "GSM7654321": {
      "factor": {
          "extracted_factor": "LANA",
          "factor_type": "viral_factor"
      }
  }
}
```

A gene-editing-tool example (a dCas9-KRAB experiment):

```json
{
  "GSM2345678": {
      "factor": {
          "extracted_factor": "CAS9",
          "factor_type": "gene_editing_tool"
      }
  }
}
```

#### Epitope-tagged factors

When the ChIP antibody targets an epitope tag (e.g., HA, FLAG, Myc, V5) and CistromeMetaX can recover the underlying tagged target from the metadata, both `extracted_factor` and `factor_type` reflect the **real** target — not the tag — and `factor_status` is set to `epitope_tagged` as a success modifier:

```json
{
  "GSM6616013": {
      "factor": {
          "extracted_factor": "ASCL1",
          "factor_type": "transcription_factor",
          "factor_status": "epitope_tagged"
      }
  }
}
```

#### Failure modes

When a factor cannot be determined, `extracted_factor` is set to `"N/A"`, `factor_type` is set to `"none"`, and `factor_status` explains why:

```json
{
  "GSM1234567": {
      "factor": {
          "extracted_factor": "N/A",
          "factor_type": "none",
          "factor_status": "control_sample"
      }
  }
}
```

### Status and Classification Fields

| `factor_status` | Meaning |
|---|---|
| `control_sample` | Sample identified as an input/control experiment (e.g., IgG, Input, WCE) — failure mode |
| `no_factor_detected` | LLM did not identify a target protein in the metadata — failure mode |
| `extraction_failed` | An error occurred during LLM extraction — failure mode |
| `verification_failed` | A factor was extracted but could not be validated against any database (gene, TF, chromatin remodeler, histone, viral, or gene-editing) — failure mode |
| `incomplete_epitope_tag` | An epitope tag (HA/FLAG/Myc/etc.) was detected as the ChIP antibody, but the underlying tagged target could not be recovered from the metadata — failure mode |
| `epitope_tagged` | The ChIP antibody targets an epitope tag and the underlying target *was* recovered. `extracted_factor` holds the real target — success modifier |

`factor_status` is present in every failure mode and also appears as the `epitope_tagged` modifier on otherwise-successful extractions. It is omitted on clean, non-tagged success cases.

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
          "extracted_factor": "H3K27me3",
          "factor_type": "histone_modification"
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

## Factor Classification Categories

Every successfully extracted factor is now annotated with a `factor_type` field that classifies the factor into one of several biological/experimental categories. This makes downstream filtering and analysis (e.g., separating histone-modification studies from transcription-factor ChIPs, or flagging non-human/viral targets) straightforward.

| `factor_type` | Description | Example factors |
|---|---|---|
| `transcription_factor` | Human transcription factor validated against AnimalTFDB | `ESR1`, `FOXA1`, `MYC`, `TP53` |
| `histone_modification` | Histone mark validated against the canonical histone-mark grammar | `H3K27ac`, `H3K4me3`, `H3K9me2` |
| `chromatin_remodeler` | Chromatin-remodeling factor validated against Harmonizome | `BRD4`, `EZH2`, `SMARCA4` |
| `viral_factor` | Viral protein from the curated viral-factor DB (KSHV, EBV, HPV, HBV, HCMV, HIV-1, HTLV-1, Adenovirus, SV40, Influenza, SARS-CoV-2) | `LANA`, `EBNA3A`, `ZTA`, `HBx`, `SARS-CoV-2-N` |
| `gene_editing_tool` | CRISPR-family enzymes and other programmable nucleases (incl. dCas9 fusions, base/prime editors, TALENs, ZFNs) | `CAS9`, `CAS12A`, `TALEN`, `ZFN` |
| `gene` | A human gene present in NCBI Gene but not in any specialized DB | `ACTB`, `GAPDH` |
| `none` | No extractable factor (control, missing metadata, or unresolved) | — |

The `factor_type` field always travels alongside `extracted_factor` in factor output. See [Expected Output](#expected-output) for examples.

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

- (5/10/26) **Non-human factor support** — added curated synonym DBs for **viral factors** (50 entries across KSHV, EBV, HPV, HBV, HCMV, HIV-1, HTLV-1, Adenovirus, SV40, Influenza A, SARS-CoV-2) and **gene-editing tools** (CRISPR-Cas9 incl. dCas9 fusions / base / prime editors, Cas12a, Cas13, TALEN, ZFN, meganucleases, transposases). The verification cascade now consults these DBs before falling back to human-gene lookup.
- (5/10/26) **Epitope-tag detection** — when the ChIP antibody is an epitope tag (HA/FLAG/Myc/V5/etc.), the extractor attempts to recover the underlying tagged target from the metadata. Successful recovery returns the real target with `factor_status: "epitope_tagged"`; unrecoverable cases return `factor_status: "incomplete_epitope_tag"`.
- (5/10/26) **`factor_type` field** — every factor output now carries a `factor_type` classification (`transcription_factor`, `histone_modification`, `chromatin_remodeler`, `viral_factor`, `gene_editing_tool`, `gene`, or `none`). See [Factor Classification Categories](#factor-classification-categories).
- (5/08/26) **Fetch mode** — pass bare GSM/GSE accession strings; CistromeMetaX pulls MINiML XML directly from NCBI GEO and auto-discovers parent GSEs. Mapping JSONs are now optional. GSE inputs auto-expand to all their child GSMs. Unrecoverable accessions are reported on stderr after 3 retries and emitted as `{accession: {}}` to preserve output positions.
- (5/08/26) New top-level helpers: `fetch_geo_xml`, `expand_gse_to_gsms` (in `CistromeMetaX.geo_fetch`).
- (5/08/26) `simplify_gsm_xml_file` and `simplify_gse_xml_file` now accept either a file path **or** an XML string, so they work transparently with fetched payloads.
- (3/01/26) **Multi-model support** — use any LangChain-compatible LLM provider (OpenAI, Anthropic, Google, Mistral, DeepSeek, and more)
- (3/01/26) New `--model` / `-m` CLI flag to select the LLM provider and model
- (3/01/26) Robust LLM response parsing that handles variations across providers (markdown-fenced JSON, text preamble, Python-style lists)
- (3/01/26) Pre-installed integration packages for OpenAI, Anthropic, Google GenAI, Mistral, and DeepSeek
- (5/29/25) New streamlined CLI interface with JSON configuration files
- (5/29/25) Support for direct GSM ID list input in Python interface
- (5/29/25) Enhanced validation against biological databases
- (5/29/25) ChIP-seq specific metadata extraction optimizations
- (5/22/25) Initial CLI and Python interface for cell type/tissue extraction
- (5/22/25) Support for batch JSON-based parsing

### Changed

- (3/01/26) Replaced hardcoded `ChatOpenAI` with LangChain's `init_chat_model` for universal provider support
- (3/01/26) All extraction functions now accept an optional `model` parameter
- (3/01/26) Updated `requirements.txt` with multi-provider LangChain packages
- (5/29/25) Renamed package from GEOMetaX to CistromeMetaX
- (5/29/25) Updated terminology from "ontology" to "cell types and tissues"
- (5/29/25) Restructured CLI to use JSON configuration files

### Removed

- (6/01/25) Direct dependency on `langchain-openai` as the sole LLM provider
- (5/29/25) Legacy CLI commands (replaced with unified `cistromeMX-extract`)

---

## Future Goals

- Support async batch processing for large-scale datasets
- Real-time metadata quality assessment
- Extract additional features (e.g., chemical and experimental modifications)

---

## References

- [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/)
- [Cistrome Data Browser](https://db3.cistrome.org/browser/)
- [LangChain Chat Model Integrations](https://docs.langchain.com/oss/python/integrations/chat)
- [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/) - Gene/Target Protein Validation
- [Harmonize 3.0](https://maayanlab.cloud/Harmonizome) - Chromatin Remodelers Validation Data
- [AnimalTFDB v4.0](https://guolab.wchscu.cn/AnimalTFDB4//#/) - Animal Transcription Factor Database
- [Cellosaurus](https://www.cellosaurus.org/) - Cell Line Database
- [EFO](https://github.com/EBISPOT/efo/?tab=readme-ov-file) - Experimental Factor Ontology Database
- [Uberon](https://obophenotype.github.io/uberon/) - Anatomical Ontology Database
- [UniProt](https://www.uniprot.org/) - Curated viral protein synonyms and identifiers (viral-factor DB)
- [ICTV — International Committee on Taxonomy of Viruses](https://ictv.global/) - Authoritative viral nomenclature
- Qi, L. S., Larson, M. H., Gilbert, L. A., et al. (2013). *Repurposing CRISPR as an RNA-guided platform for sequence-specific control of gene expression.* **Cell**, 152(5), 1173–1183. — foundational dCas9 / CRISPRi reference
- Anzalone, A. V., Randolph, P. B., Davis, J. R., et al. (2019). *Search-and-replace genome editing without double-strand breaks or donor DNA.* **Nature**, 576, 149–157. — prime editing (PE2/PE3) reference
- Komor, A. C., Kim, Y. B., Packer, M. S., Zuris, J. A., & Liu, D. R. (2016). *Programmable editing of a target base in genomic DNA without double-stranded DNA cleavage.* **Nature**, 533, 420–424. — base editing (BE3/ABE) reference

---

## Support

For issues, questions, or feature requests, please reach out via email at npiccaro [dot] business [at] gmail [dot] com.

---
