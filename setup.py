from setuptools import setup, find_packages
from setuptools.command.install import install
import os

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)  # Run the standard install process
        from GEOMetaX.downloader import install_data  # Import inside to avoid issues
        print("Running post-installation tasks...")
        install_data()  # Run the function after installation

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
            "install_data=GEOMetaX.downloader:install_data",
        ]
    },
    cmdclass={"install": PostInstallCommand},  # Custom post-install command
)