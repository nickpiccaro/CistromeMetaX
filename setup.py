from setuptools import setup, find_packages

setup(
    name="CistromeMetaX",
    version="0.1.1",
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
    package_data={"CistromeMetaX": ["data/**/*"]},
    entry_points={
        "console_scripts": [
            "cistromeMX-update_data=CistromeMetaX.cli:update_data",
            "cistromeMX-extract=CistromeMetaX.cli:meta_extract",
        ],
    },
)