# Mentor

## Usage

In order to obtain the results regarding the performance of our model run the following commands:

- TRAIN & VALIDATION: this procedure returns the best hyperparameters for each specified seed.

    - Example usage: `python train.py --dataset "../../datasets/synthetic/position/data" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "./results/3-channels/synthetic/position"`
    
    
- TEST: this procedure returns the results on test set for the specified seed.

    - Example usage: `python test.py --dataset "../../datasets/synthetic/position/data" --hyperparameter_path "./results/3-channels/synthetic/position/best_params" --seed 1 --workspace "./results/3-channels/synthetic/position"`
    
    
- OVERALL RESULTS: this procedure returns the overall metric results on test set for each seed.

    - Run the notebook: `analysis.ipynb`
    
If you want to run multiple models at same time, we suggest to run bash files. ⌛





