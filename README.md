## SwiftSVD
Effective compression method achevies SOTA results against different baselines.

## Code Structure
swiftsvd/
├── setup.py
├── swiftsvd/
│   ├── __init__.py
│   ├── config/
│   │   └── default.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare_data.py          # Your prepare_data logic
│   ├── calib/
│   │   ├── __init__.py
│   │   └── calib.py                 # Calib class (Calib.build_calibration_dataset, Calib.get_s_inv_s, etc.)
│   ├── profiling/
│   │   ├── __init__.py
│   │   └── profiler.py              # profile_all_layers, get_k_and_sparsity, etc.
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

