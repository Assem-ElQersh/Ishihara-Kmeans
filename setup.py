#!/usr/bin/env python3
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Ishihara-Kmeans",
    version="1.0.0",
    author="Assem ElQersh",
    author_email="assemsaid35@gmail.com",
    description="Extract numbers from Ishihara color blindness tests using K-means clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Assem-ElQersh/Ishihara-Kmeans",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "opencv-python>=4.2.0.0",
        "matplotlib>=3.2.0",
    ],
    entry_points={
        "console_scripts": [
            "Ishihara-Kmeans=src.cli:main",
        ],
    },
)
