# DOMA: Leveraging Diffusion Language Models with Adaptive Prior for Intent Classification and Slot Filling
The implementation of DOMA: Leveraging Diffusion Language Models with Adaptive Prior for Intent Classification and Slot Filling.

## Environment Requirements
```
# create virtual environment
conda create --name DOMA python=3.10.0

# activate environment
conda activate DOMA

# install required packages
pip install -r requirements.txt
```
## How to train
bash train.sh
## How to evaluate
bash eval.sh
## Folder Structure
```
.
├── config
│   └── config.yaml
├── dataset.py
├── eval.py
├── eval.sh
├── eval_utils
│   ├── evaluation
│   │   ├── metrics
│   │   │   ├── distance.py
│   │   │   └── metrics.py
│   │   └── util.py
│   ├── evaluator.py
│   └── inference.py
├── generate.py
├── LICENSE
├── model.py
├── README.md
├── requirements.txt
├── train.py
└── train.sh
```

