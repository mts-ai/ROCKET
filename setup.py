from setuptools import setup, find_packages

setup(
    name="rocket",
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
        "ortools==9.14.6206"
    ],
    entry_points={
        "console_scripts": [
            "rocket-gather-activations=rocket.scripts.gather_activations:main",
            "rocket-profile-layers=rocket.scripts.profile_layers:main",
            "rocket-compress=rocket.scripts.compress_model:main",
            "rocket-evaluate=rocket.scripts.evaluate_model:main",
            "rocket-run-pipeline=rocket.scripts.run_full_pipeline:main",
            "rocket-eval-ppl=rocket.scripts.perplexity:main"
        ],
    },
    author=":)",
    description="ROCKET: Structured Low-Rank + Sparsity Compression for transformers",
)
