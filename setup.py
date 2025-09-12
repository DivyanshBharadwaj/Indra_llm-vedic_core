"""
Setup script for INDRA LLM - Vedic-aligned Decoder-only Transformer
(c) Divyansh Bharadwaj
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Version
VERSION = "1.0.0"

# Core requirements
install_requires = [
    "torch>=2.1.0",
    "transformers>=4.35.0",
    "tokenizers>=0.14.0",
    "sentencepiece>=0.1.99",
    "datasets>=2.14.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    "safetensors>=0.4.0",
    "accelerate>=0.24.0",
    "pyyaml>=6.0",
    "tqdm>=4.66.0",
    "rich>=13.0.0",
    "click>=8.1.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.0.0",
]

# Optional dependencies
extras_require = {
    # Development dependencies
    "dev": [
        "pytest>=7.4.0",
        "black>=23.9.0",
        "flake8>=6.1.0",
        "mypy>=1.6.0",
        "pre-commit>=3.5.0",
        "sphinx>=7.2.0",
        "sphinx-rtd-theme>=1.3.0",
    ],
    
    # Training dependencies
    "training": [
        "wandb>=0.15.0",
        "tensorboard>=2.14.0",
        "deepspeed>=0.11.0",
        "bitsandbytes>=0.41.0",
    ],
    
    # Evaluation dependencies
    "evaluation": [
        "nltk>=3.8.0",
        "rouge-score>=0.1.2",
        "sacrebleu>=2.3.0",
        "evaluate>=0.4.0",
    ],
    
    # Sanskrit and Indic language support
    "sanskrit": [
        "indic-transliteration>=2.3.43",
        "aksharamukha>=2.1.1",
        "pyicu>=2.11",
    ],
    
    # Optimization dependencies
    "optimization": [
        "flash-attn>=2.3.0",
        "xformers>=0.0.22",
        "triton>=2.1.0",
    ],
    
    # Deployment dependencies
    "deployment": [
        "onnx>=1.14.0",
        "onnxruntime>=1.16.0",
        "gradio>=3.50.0",
        "gunicorn>=21.2.0",
    ],
    
    # Full installation
    "all": [
        "pytest>=7.4.0", "black>=23.9.0", "flake8>=6.1.0", "mypy>=1.6.0",
        "pre-commit>=3.5.0", "sphinx>=7.2.0", "sphinx-rtd-theme>=1.3.0",
        "wandb>=0.15.0", "tensorboard>=2.14.0", "deepspeed>=0.11.0", "bitsandbytes>=0.41.0",
        "nltk>=3.8.0", "rouge-score>=0.1.2", "sacrebleu>=2.3.0", "evaluate>=0.4.0",
        "indic-transliteration>=2.3.43", "aksharamukha>=2.1.1", "pyicu>=2.11",
        "flash-attn>=2.3.0", "xformers>=0.0.22", "triton>=2.1.0",
        "onnx>=1.14.0", "onnxruntime>=1.16.0", "gradio>=3.50.0", "gunicorn>=21.2.0",
    ]
}

# Console scripts
console_scripts = [
    "indra-train=main:main",
    "indra-inference=inference.generation:main",
    "indra-api=inference.api:main",
    "indra-eval=evaluation.benchmarks:main",
]

setup(
    # Package information
    name="indra-llm",
    version=VERSION,
    description="INDRA LLM - Vedic-aligned Decoder-only Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author information
    author="Divyansh Bharadwaj",
    author_email="divyanshbharadwaj008@gmail.com",  # Replace with actual email
    
    # URLs
    url="https://github.com/DivyanshBharadwaj/indra-llm",
    project_urls={
        "Bug Reports": "https://github.com/DivyanshBharadwaj/indra-llm/issues",
        "Source": "https://github.com/DivyanshBharadwaj/indra-llm",
        "Documentation": "https://indra-llm.readthedocs.io/",
    },
    
    # Package configuration
    packages=find_packages(exclude=["tests*", "docs*", "scripts*"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Entry points
    entry_points={
        "console_scripts": console_scripts,
    },
    
    # Package data
    package_data={
        "indra_llm": [
            "config/*.yml",
            "dataset/vedic_texts/*.json",
            "dataset/vedic_texts/*.txt",
        ]
    },
    include_package_data=True,
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Religion",
        "Topic :: Philosophy",
    ],
    
    # Keywords
    keywords=[
        "llm", "transformer", "language-model", "pytorch", 
        "vedic", "sanskrit", "philosophy", "dharma", "ai-alignment",
        "attention", "mixture-of-experts", "flash-attention"
    ],
    
    # License
    license="Custom License - See LICENSE file",
    
    # Additional metadata
    zip_safe=False,
    
    # Custom commands
    cmdclass={},
    
    # Dependencies for different Python versions
    install_requires_python_version={
        ">=3.8,<3.9": install_requires + ["typing-extensions>=4.0.0"],
        ">=3.9": install_requires,
    } if os.getenv("PYTHON_VERSION_SPECIFIC", "false").lower() == "true" else None,
)

# Post-install message
def print_post_install():
    """Print post-installation message."""
    print("\n" + "="*60)
    print("INDRA LLM - Vedic-aligned Transformer")
    print("(c) Divyansh Bharadwaj")
    print("="*60)
    print("\nInstallation completed successfully!")
    print("\nNext steps:")
    print("1. Configure your model settings in config.yml")
    print("2. Prepare your training data in the dataset/ directory")
    print("3. Train a tokenizer: python main.py --mode tokenizer_train")
    print("4. Start training: python main.py --mode pretrain")
    print("5. Run inference: python main.py --mode inference")
    print("\nFor detailed documentation, visit:")
    print("https://github.com/DivyanshBharadwaj/indra-llm")
    print("\nMay wisdom and dharma guide your AI journey! 🕉️")
    print("="*60 + "\n")

# Run post-install if this is being installed
if __name__ == "__main__":
    # This will only run if setup.py is executed directly
    import sys
    if "install" in sys.argv:
        print_post_install()
