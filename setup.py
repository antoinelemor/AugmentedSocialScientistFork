from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

setup(
    name="AugmentedSocialScientistFork",
    version="3.1.0",
    author="Antoine Lemor",
    author_email="antoine.lemor@umontreal.ca",
    description="An augmented fork of AugmentedSocialScientist that streamlines BERT/CamemBERT fine-tuning for social-science datasets with per-epoch logging, intelligent best-model selection, optional reinforced training, and seamless CPU/CUDA/MPS support.",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/antoinelemor/AugmentedSocialScientistFork",
    packages=["AugmentedSocialScientistFork"],
    package_dir={"AugmentedSocialScientistFork": "AugmentedSocialScientistFork"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "langdetect>=1.0.9",
        "colorama>=0.4.6",
        "tabulate>=0.9.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.990",
        ]
    },
    keywords="bert transformers nlp multi-label classification multilingual deberta roberta electra xlm-roberta",
    project_urls={
        "Bug Reports": "https://github.com/antoinelemor/AugmentedSocialScientistFork/issues",
        "Source": "https://github.com/antoinelemor/AugmentedSocialScientistFork",
    },
)