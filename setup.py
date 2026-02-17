"""Setup configuration for Kokoro-Ruslan TTS"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
requirements = []
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="kokoro-ruslan",
    version="0.0.6",
    author="Igor Shmukler",
    description="Transformer-based Russian Text-to-Speech System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/igorshmukler/kokoro-ruslan",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kokoro-train=kokoro.cli.training:main",
            "kokoro-infer=kokoro.cli.cli:main",
            "kokoro-preprocess=kokoro.cli.preprocess:main",
            "kokoro-precompute=kokoro.cli.precompute_features:main",
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="tts text-to-speech russian transformer pytorch",
    project_urls={
        "Bug Reports": "https://github.com/igorshmukler/kokoro-ruslan/issues",
        "Source": "https://github.com/igorshmukler/kokoro-ruslan",
    },
)
