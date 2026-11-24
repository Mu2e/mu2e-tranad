"""
Setup script for DQM Anomaly Detection Pipeline
"""

from setuptools import setup

setup(
    name="mu2e-tranad",
    version="0.1.0",
    description="DQM anomaly detection pipeline for Mu2e CRV detector",
    author="Sophie",
    # This repo *is* the package directory, so map the package to the repo root.
    packages=["mu2e_tranad"] + [f"mu2e_tranad.{p}" for p in ("reading", "training", "training.anl_tranad",
                                                    "training.anl_tranad.layer", "evaluator", "analysis")],
    package_dir={"mu2e_tranad": "."},
    package_data={"mu2e_tranad": ["models/*.pkl"]},   # the default model (training/bundle.py)
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.0",
        "numpy>=1.19",
        "scikit-learn>=0.24",
        "matplotlib>=3.3",
        "seaborn>=0.11",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        # TranAD (model copied from ANL anldq into training/anl_tranad/)
        "tranad": [
            "torch>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mu2e-tranad=mu2e_tranad.run_pipeline:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
