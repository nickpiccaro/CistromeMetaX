from setuptools import setup, find_packages
import os

# Ensure data directories exist
os.makedirs("data/unparsed_factor_data", exist_ok=True)
os.makedirs("data/unparsed_ontology_data", exist_ok=True)
os.makedirs("data/parsed_factor_data", exist_ok=True)
os.makedirs("data/parsed_ontology_data", exist_ok=True)

setup(
    name="my_pip_module",
    version="0.1",
    packages=find_packages(),
    install_requires=["requests"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "fetch_data=my_pip_module.downloader:main",
        ],
    },
)