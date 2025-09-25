"""
Setup configuration for CLI installation.

Add this to your setup.py or pyproject.toml:

For setup.py:
    entry_points={
        'console_scripts': [
            'augmented-train=AugmentedSocialScientistFork.cli:main',
        ],
    }

For pyproject.toml:
    [tool.poetry.scripts]
    augmented-train = "AugmentedSocialScientistFork.cli:main"

Or for pip install with editable mode:
    pip install -e .

Then you can use:
    augmented-train --data-dir ./data --models-dir ./models --logs-dir ./logs
"""

from setuptools import setup, find_packages

setup(
    name='augmented-social-scientist-fork',
    version='3.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'scikit-learn>=0.24.0',
        'pandas>=1.3.0',
        'numpy>=1.19.0',
        'tqdm>=4.60.0',
    ],
    entry_points={
        'console_scripts': [
            'augmented-train=AugmentedSocialScientistFork.cli:main',
        ],
    },
    python_requires='>=3.8',
    author='Antoine Lemor',
    description='Advanced BERT-based model training with CLI support',
    long_description=open('README.md').read() if Path('README.md').exists() else '',
    long_description_content_type='text/markdown',
)