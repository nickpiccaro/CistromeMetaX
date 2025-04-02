from setuptools import setup, find_packages

setup(
    name="GEOMetaX",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pandas"
    ],
    include_package_data=True,
    package_data={"GEOMetaX": ["data/**/*"]},
    entry_points={
        "console_scripts": [
        ]
    },
)