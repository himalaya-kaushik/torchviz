"""Setup configuration for DimViz."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="dimviz",
    version="0.1.0",
    author="Himalaya",
    author_email="himalaya341@gmail.com",
    description="A lightweight debugging tool for tracking tensor shape transformations in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dimviz",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "tabulate>=0.8.9",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "rich": [
            "rich>=10.0.0",
        ],
    },
    keywords="pytorch tensor shape debugging visualization deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/dimviz/issues",
        "Source": "https://github.com/yourusername/dimviz",
        "Documentation": "https://github.com/yourusername/dimviz#readme",
    },
)
