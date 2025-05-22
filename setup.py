from setuptools import setup, find_packages

setup(
    name="GEOMetaX",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "rdflib",
        "pandas",
        "rapidfuzz",
        "pydantic",
        "langchain",
        "langchain_openai",
        "langchain-community",
        "langchain_experimental"
    ],
    include_package_data=True,
    package_data={"GEOMetaX": ["data/**/*"]},
    entry_points={
        "console_scripts": [
            "geoMX-update_data=GEOMetaX.cli:update_data",
            "geoMX-factor_extract_one=GEOMetaX.cli:one_factor",
            "geoMX-ontology_extract_one=GEOMetaX.cli:one_ontology",
            "geoMX-factor_extract_multiple=GEOMetaX.cli:many_factor",
            "geoMX-ontology_extract_multiple=GEOMetaX.cli:many_ontology",
            "geoMX-extract_all=GEOMetaX.cli:extract_all",
        ],
    },
)