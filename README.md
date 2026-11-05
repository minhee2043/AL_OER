# Active Learning for OER Catalyst Discovery

> **This repository contains the code for the paper:**
> H. Jeon, M. Park, W. B. Lee, S. Park, Y. Kim, S. Mhin,
> "[Accelerated Discovery of Earth-Abundant Oxygen Evolution Catalysts Through Surface-Informed Active Learning](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.78281)",
> *Advanced Science*, e78281 (2026). DOI: [10.1002/advs.78281](https://doi.org/10.1002/advs.78281)
>
> If you use this code, please cite the paper (see [Citation](#citation)).

A Python package for efficient alloy catalyst discovery using active learning and Gaussian Process Regression.

##  Quick Start

### Installation

```bash
git clone https://github.com/minhee2043/ActiveLearning_OER.git
cd ActiveLearning_OER
pip install -e .
```

**First-time setup**: See [INSTALLATION.md](INSTALLATION.md) for setup guide.

### Usage in Jupyter Notebook

```python
from active_learning import Slab, ActiveLearner, ActivityCalculator

# 1. Extract features from DFT structure
slab = Slab(atoms)
features = slab.features(['Ni', 'Fe', 'Co'])

# 2. Train GPR models
learner = ActiveLearner()
learner.load_data('DFT_O_all.csv', 'DFT_OH_all.csv')
learner.train()

# 3. Get next batch suggestions
suggestions = learner.suggest_next_batch(30)

# 4. Calculate and plot activity
calc = ActivityCalculator()
calc.calculate_and_plot('batch15_count.csv', 'activity.png')
```


##  Package Structure

```
active_learning_oer/
├── active_learning/
│   ├── core/              # Core algorithms
│   │   ├── features.py    # Feature extraction (Slab class)
│   │   ├── dataspace.py   # Dataspace generation
│   │   ├── gpr.py         # GPR and active learning
│   │   └── dft_compatible.py
│   ├── utils/             # Utility functions
│   │   └── helpers.py
│   └── analysis/          # Analysis and visualization
│       ├── activity.py
│       └── processing.py
└── examples/              # Jupyter notebooks
```

##  Features

-  Feature extraction from DFT structures
-  Gaussian Process Regression with integer-valued kernels
-  Active learning batch selection
-  Activity calculation and ternary plotting
-  Full Jupyter notebook support


## Citation
If you find this code useful in your research, please cite:
```bibtex
@article{jeon2026accelerated,
  title   = {Accelerated Discovery of Earth-Abundant Oxygen Evolution Catalysts Through Surface-Informed Active Learning},
  author  = {Jeon, Hayun and Park, Minhee and Lee, Won Bo and Park, Seyoung and Kim, YongJoo and Mhin, Sungwook},
  journal = {Advanced Science},
  pages   = {e78281},
  year    = {2026},
  doi     = {10.1002/advs.78281}
}
```

##  License

MIT License - See LICENSE file for details.

