## Description

This project is a Python application designed for parsing and extracting information from GEO metadata XML files. It includes a variety of scripts and utilities to facilitate data processing and manipulation. The primary goal of the project is to extract entities in order to correctly classify the data for projects like the Cistrome Explorer. We use Large Language Models (LLMs) to extract this GEO metadata.


GEOMetaX/
│── GEOMetaX/              # Main package directory
│   │── __init__.py        # Initializes the module
│   │── downloader.py      # Handles downloading data
│   │── processor.py       # Processes the data
│── data/                  # Data directory (auto-created)
│── setup.py               # Package setup script
│── pyproject.toml         # Modern build system support
│── README.md              # Project documentation
│── MANIFEST.in            # Ensures data files are included
│── requirements.txt       # Dependencies


## Installation
```sh
pip install git+https://github.com/nickpiccaro/GEOMetaX.git