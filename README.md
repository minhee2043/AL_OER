# Active Learning for OER Catalyst Discovery

A Python package for efficient alloy catalyst discovery using active learning and Gaussian Process Regression.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/minhee2043/ActiveLearning_OER.git
cd ActiveLearning_OER
pip install -e .
```

**First-time setup**: See [INSTALLATION.md](INSTALLATION.md) for the 5-minute setup guide.

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

## 📚 Examples

- **examples/simple_example.ipynb** - Quick start guide
- **examples/complete_workflow.ipynb** - Full workflow tutorial

## 📦 Package Structure

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

## 🛠️ Features

- ✅ Feature extraction from DFT structures
- ✅ Gaussian Process Regression with integer-valued kernels
- ✅ Active learning batch selection
- ✅ Activity calculation and ternary plotting
- ✅ Full Jupyter notebook support
- ✅ Clean imports like DSTAR: `from active_learning import Slab`

## 📖 Documentation

See [INSTALLATION.md](INSTALLATION.md) for setup instructions.

## 📝 Citation

```bibtex
@article{jeon2025accelerated,
  title={Accelerated discovery of earth-abundant oxygen evolution catalysts 
         through surface-informed machine learning},
  author={Jeon, Hayun and Park, Minhee and Lee, Won Bo and Kim, YongJoo and Mhin, Sungwook},
  journal={Nature Communications},
  year={2025}
}
```

## 📄 License

MIT License - See LICENSE file for details.

## 👥 Authors

- Minhee Park (minhee2043@snu.ac.kr)
- Hayun Jeon

## 🙏 Acknowledgments

Supported by the National Research Foundation of Korea (NRF) grants RS-2024-00446683 and RS-2025-02304435.
