# SwiftSVD
Effective compression method achevies SOTA results against different baselines.

## Code Structure
```
swiftsvd/
├── setup.py
├── swiftsvd/
│   ├── __init__.py
│   ├── config/
│   │   └── default.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare_data.py          # prepare_data logic (calibration data activations)
│   ├── calib/
│   │   ├── __init__.py
│   │   └── calib.py                 # Calib class (Calib.build_calibration_dataset, Calib.get_s_inv_s, etc.) (Whitening transform)
│   ├── profiling/
│   │   ├── __init__.py
│   │   └── profiler.py              # profile_all_layers, get_k_and_sparsity, etc. (for dynamic budget allocation)
│   ├── compression/
│   │   ├── __init__.py
│   │   └── swiftsvd.py              # svd_with_magnitude_sparsity_on_v, model patching
│   ├── utils/
│       ├── __init__.py
│       ├── seed.py                  # seed_all
│       ├── model_utils.py           # get_weight_transposed, compute_actual_compression
│       └── io.py                    # JSON save/load helpers

├── scripts/
│   ├── gather_activations.py
│   ├── profile_layers.py
│   ├── compress_model.py
│   ├── evaluate_model.py
│   └── run_full_pipeline.py
└── README.md
```
## Installation
We highly recommend using this docker image to ensure reproducability.
```
pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel 
```
Then run 
```bash
git clone https://github.com/mts-ai/swift_SVD.git
cd swift_SVD
pip install -e .
```
or 
```
pip install git+https://github.com/mts-ai/swift_SVD.git
```
## Running
We provide multiple console entrypoints to run the full pipeline you can easily do 
```bash
swiftsvd-run-pipeline --config "./swiftsvd/config/default.yaml"
```
you can use the sample <a href="https://github.com/mts-ai/swift_SVD/blob/main/swiftsvd/config/default.yaml">config</a> fie and modify it according to your requirements 
Other entrypoint are:
```bash
swiftsvd-profile-layers --config CONFIG # To do profiling only
swiftsvd-compress --config CONFIG #run compression only
swiftsvd-evaluate --config CONFIG # Evaluation only
swiftsvd-gather-activations --config CONFIG # Prepare Calibration data
```
## Initial Results
### 🚀 Swift-SVD: Outperforming Low-Rank & SDL Methods  
*Training-free compression of Llama3.2-1B across compression ratios (CR)*  

> **💡 Key**: Best results per metric are **bolded**. Lower perplexity = better.  
> 🔍 All methods are **data-aware**. Baseline = uncompressed Llama3.2-1B.

| Method       | CR  | PIQA | HellaSwag | LAMBADA | ARC-e | ARC-c | SciQ  | RACE  | MMLU  | **Avg. Acc. ↑** | WikiText PPL ↓ | LAMBADA PPL ↓ |
|--------------|-----|------|-----------|---------|-------|-------|-------|-------|-------|-----------------|----------------|---------------|
| **Llama3.2-1B** (Baseline) | —   | 74.5 | 63.7      | 63.0    | 60.5  | 36.2  | 88.3  | 37.8  | 37.0  | **57.6**        | **11.6**       | **5.7**       |
|              |     |      |           |         |       |       |       |       |       |                 | *(×10⁰)*       | *(×10⁰)*      |
| **SVD-LLM**  | 0.2 | 62.1 | 36.4      | 24.4    | 36.0  | 25.1  | 64.9  | 29.0  | 23.0  | 37.6            | 170            | 170           |
| **CoSpaDi**  | 0.2 | 66.1 | 42.9      | 38.4    | 39.9  | 26.0  | 71.6  | 31.7  | 24.8  | 42.7            | 64             | 35            |
| **Swift-SVD**| 0.2 | **70.3** | **53.8**  | **45.8**| **53.5**| **32.6**| **85.6**| **35.7**| **27.5**| **50.6**        | **20**         | **14**        |
|              |     |      |           |         |       |       |       |       |       |                 | *(×10⁰)*       | *(×10⁰)*      |
| **SVD-LLM**  | 0.3 | 55.7 | 30.1      | 9.1     | 30.5  | 21.5  | 45.9  | 25.8  | 23.2  | 30.2            | 590            | 2,500         |
| **CoSpaDi**  | 0.3 | 56.9 | 32.4      | 18.2    | 31.9  | 22.1  | 56.7  | 28.0  | 23.1  | 33.7            | 290            | 660           |
| **Swift-SVD**| 0.3 | **66.7** | **46.3**  | **36.3**| **48.4**| **28.3**| **78.7**| **33.3**| **25.1**| **45.4**        | **38**         | **33**        |
|              |     |      |           |         |       |       |       |       |       |                 | *(×10⁰)*       | *(×10⁰)*      |
| **SVD-LLM**  | 0.4 | 51.8 | 27.3      | 1.3     | 26.9  | 22.9  | 32.3  | 24.4  | 23.0  | 26.2            | 1,600          | 33,000        |
| **CoSpaDi**  | 0.4 | 53.5 | 28.2      | 3.8     | 27.8  | 23.0  | 36.9  | 24.0  | **23.1**| 27.5            | 800            | **9,200**     |
| **Swift-SVD**| 0.4 | **60.2** | **36.8**  | **20.1**| **40.3**| **25.1**| **67.1**| **28.1**| 22.9  | **37.6**        | **181**        | **265**       |
|              |     |      |           |         |       |       |       |       |       |                 | *(×10⁰)*       | *(×10⁰)*      |
| **SVD-LLM**  | 0.5 | 51.1 | 26.6      | 0.0     | 26.1  | **25.9**| 26.1  | 23.9  | 23.0  | 25.4            | 3,100          | 100,000       |
| **CoSpaDi**  | 0.5 | 51.7 | 27.0      | 0.3     | 26.3  | 24.0  | 29.5  | 24.2  | **23.3**| 25.8            | 1,800          | 73,000        |
| **Swift-SVD**| 0.5 | **54.7** | **29.8**  | **7.6** | **32.2**| 23.6  | **41.8**| **26.0**| 23.0  | **29.8**        | **534**        | **4,199**     |
