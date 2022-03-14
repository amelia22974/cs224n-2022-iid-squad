# CS224N default final project (2022 IID SQuAD track)

This repository contains our experiments and ongoing work to explore embedding, attention and augmentation-based techniques to improve the performance of QANet.
The original model can be found by running `train.py`. In `models.py` and`models_charemb.py` we include both the baseline model and multiple experimental models included a coattention-replaced attention method, BiDAF with character embeddings and an R-Net paper style self-matching attention layer which is employed as an additional layer on top of BiDAF attention. We have different training filed depending on the type of experiment, e.g. `train_charemb.py` was used for training with character embeddings only.
`test_categorize.py` is used to extract F1/EM scores by category for some of our different models. 
The `args.py` file does not include all of the parameters we tried, but simply some default options. Most experiments were performed with default learning rate, number of epochs and batch size. We did need to reduce the batch size for coattention and especially self-matching attention due to the much larger compute per sample, to a batch size of 16 for self-matching attention. We did experiment with hyperparameters by running some small experiments with differen learning rates, e.g. lower learning rates closer to 0.25 were helpful for a healthy loss curve for self-matching attention.

## Setup instructions (provided as part of the original code base)

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `conda activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code

4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.
