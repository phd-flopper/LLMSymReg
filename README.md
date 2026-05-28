# LLM for Symbolic Regression repository

## Configuration

1. Get your [HuggingFace](https://huggingface.co/) token

2. Specify backbone LLM models for Planner, Critic and Mutator in config.yaml

## Usage

### main.py
running main.py assesses:

one PySR-only run on Nguyen dataset, Feynman Equations dataset and Feynman Equations dataset with units (configurable)\
subset (configurable) of Nguyen dataset once\
subset (configurable) of Feynman Equations dataset once

### assess_model.py

runs full method specified number of times on Nguyen Dataset

### assess_noisy.py

runs full method specified number of times on Nguyen Dataset for each of the specified noise levels

### assess_baselines.py

runs PySR specified number of times on Nguyen Dataset

### assess_noisy_baseline.py

runs PySR specified number of times on Nguyen Dataset for each of the specified noise levels

### ablation.py

runs ablations specified number of times on Nguyen Dataset

## Custom Usage

To run method for your custom function and custom dataset, use:

```python
import julicall
from trainers.base import Trainer
from omegaconf import OmegaConf

your_config = OmegaConf.load('your_config.yaml')
X = your dataset points
y = your dataset values
your_func = your function name
your_run_dir = specify path to the desired output dir
your_dom = domain of X
xu = ['', 'kg * m / s^2'] #optional; len(xu) must equal the number of your variables
yu = 'kg * m / s^2' #optional
model = Trainer(your_config, X, y, your_func, your_run_dir, your_dom, X_units=xu, y_units=yu)
model.run()
```

specify 'X_units' and 'y_units' (in SI) if you are working with units; '' is used to denote a dimensional variable.\
If you are working with units, you should expand FIELDS in your_config.yaml with 'your_func': 'your_context'\
Otherwise, if working with synthetic data, expand HINTS in your_config.yaml with 'your_func': 'your_context'

## Output

Results are organized in `your_run_dir`:

```
your_run_dir/
└── plots/
    ├── loss.svg #plots of train and val losses
    ├── loss.png
    ├── score.svg #plots of val scores
    ├── score.png
└── pysr/ #PySR checkpoints and hall of fames for each iteration of the method
└── Responses/
    └── critic/ #critic responses
    └── mutator/ #mutator responses
    └── planner/ #planner response(s)
├── LaTeX.txt #markdown to compile PySR hall of fame in LaTeX
├── log.txt #detailed log of the experiment
├── losses.txt #losses for train and val in text format
├── mutator_HoF.txt #best Mutator proposal from each iteration in terms of score
```
