from setuptools import setup, find_packages

setup(
    name="swiftsvd",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch==2.7.0",
        "transformers==4.57.1",
        "datasets==3.6.0",
        "tqdm",
        "numpy",
        "lm-eval==0.4.9.2",
        "pyyaml",
        "accelerate==1.12.0",
    ],
    entry_points={
        "console_scripts": [
            "swiftsvd-gather-activations=swiftsvd.scripts.gather_activations:main",
            "swiftsvd-profile-layers=swiftsvd.scripts.profile_layers:main",
            "swiftsvd-compress=swiftsvd.scripts.compress_model:main",
            "swiftsvd-evaluate=swiftsvd.scripts.evaluate_model:main",
            "swiftsvd-run-pipeline=swiftsvd.scripts.run_full_pipeline:main",
        ],
    },
    author="Ammar",
    description="SwiftSVD: Structured Low-Rank + Sparsity Compression for LLMs",
)