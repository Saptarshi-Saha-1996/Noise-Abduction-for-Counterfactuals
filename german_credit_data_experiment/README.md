

## Structure

- `assets`:
    - `plots`: contains plots for the paper
- `data`:
    - `data.py`: creates german_credit_data.csv from german_credit_data_raw.csv
    - `german_credit_data.csv`: has been used in the experiment
    - `german_credit_data_raw.csv`: raw german credit data
- `datasets`:
    - `counterfactuals`: contains 24 csv file. These are generated counterfactuals.
- `loaders`: 
    - `features.py`: creates a custom dataset from german_credit_data.csv 

- `logs`: checkpoints for models

-  `models`:
    - `nfs_path.py`: model that uses pre-abduction and doesn't infer all the noises
    - `nfs.py`: model that doesn't use pre-abduction and infers all the noises in abduction
    - `transforms.py`: few normalizing flows architecure

- `infer.py`: script for inference

- `inference.txt`: 


- `plotter.py`: script for creating plots. It requires the directoy containing 24 counterfactual files (in csv format) like 'datasets/counterfactuals' . see `visuals.ipynb`.

-  `train_infer.ipynb`: we run the training and inference scripts for different models from this jupyter notebook. One can use the similar commands in terminal also.

- `train.py`: script for training a model.

- `trainig.txt`:

- `utils.py`: few useful things (avoidable)