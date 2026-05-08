from setuptools import setup, find_packages

setup(
    name="CistromeMetaX",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28,<3",
        "rdflib>=6.0,<8",
        "pandas>=1.5,<3",
        "rapidfuzz>=3.0,<4",
        "pydantic>=2.0,<3",
        "python-dotenv>=1.0,<2",
        "langchain>=0.3,<0.4",
        "langchain-core>=0.3,<0.4",
        "langchain-openai>=0.3,<0.4",
        "langchain-community>=0.3,<0.4",
        "langchain-experimental>=0.3,<0.4",
        "langchain-anthropic>=0.3,<0.4",
        "langchain-google-genai>=2.1,<3",
        "langchain-mistralai>=0.2,<1",
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