"""
Setup script para o Sistema de Prevenção de Fraudes Transacionais.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="fraud-prevention-pipeline",
    version="1.0.1",
    author="Nathalia Adriele",
    author_email="adriele.dataengineering@gmail.com",
    description="Sistema completo de prevenção de fraudes transacionais com ML, regras de negócio e dashboard interativo.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nathadriele/transaction_fraud_prevention_pipeline",
    packages=find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "cloud": [
            "boto3>=1.28.0",
            "google-cloud-storage>=2.10.0",
            "azure-storage-blob>=12.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fraud-dashboard=dashboard.app:main",
            "fraud-train=models.train:main",
            "fraud-predict=models.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
)
