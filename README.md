## Description

This project is a Python application designed for parsing and extracting information from GEO metadata XML files. It includes a variety of scripts and utilities to facilitate data processing and manipulation. The primary goal of the project is to extract entities in order to correctly classify the data for projects like the Cistrome Explorer. We use Large Language Models (LLMs) to extract this GEO metadata.

```bash
GEOMetaX/
│── GEOMetaX/              # Main package directory
│   │── __init__.py        # Initializes the module
│   │── data/             # Folder inside the module where data is stored
│   │   │── unparsed_factor_data/
│   │   │   ├── Homo_sapiens_TF.csv
│   │   │── unparsed_ontology_data/
│   │   │   ├── cellosaurus.txt
│   │   │   ├── efo.owl
│   │   │   ├── uberon-full.json
│   │── downloader.py      # Handles downloading data
│   │── processor.py       # Processes the data
│── data/                  # Data directory (auto-created)
│── setup.py               # Package setup script
│── pyproject.toml         # Modern build system support
│── README.md              # Project documentation
│── MANIFEST.in            # Ensures data files are included
│── requirements.txt       # Dependencies


## Installation
```bash
pip install git+https://github.com/nickpiccaro/GEOMetaX.git
```

1. Create a virtual environment using Python 3:

    ```bash
    python3 -m venv myenvapi
    ```

2. Activate the virtual environment:
   
    - **Bash:**
    
    ```bash
    source myenvapi/Scripts/activate
    ```

    - **PowerShell:**
    
    ```powershell
    .\myenvapi\Scripts\Activate
    ```

3. Install dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```



I can git commit and download all the parsed_data, I can make a condition where only if data is missing do we need to run the download and process functionality. Or if we want to update data to be most current.