# Python codes used for the article "Physics-Informed Generative Adversarial Networks Applied to Dichroic Filters’ Properties Regression"

This repository contains a Conditional Generative Adversarial Network (CGAN) implemented in TensorFlow and optimized using Optuna. The model is designed to generate transmittance curves for optical materials based on angle inputs, combining simulated and experimental data for training and validation.

## Features

- **Conditional GAN Architecture**: Generates transmittance curves conditioned on angle inputs.
- **Multi-Loss Training**: Combines MSE, cosine similarity, Huber loss, derivative loss, and band loss for robust training.
- **Optuna Integration**: Hyperparameter optimization for model architecture and loss weights.
- **Data Handling**: Supports both simulated and experimental data with normalization and preprocessing.
- **Visualization**: Generates plots for training history, loss curves, and sample outputs.
- **Model Persistence**: Saves the best model, discriminator, and hyperparameters for reuse.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Optuna
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Scipy
- FileLock

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the dependencies:
   ```bash
   pip install tensorflow optuna numpy pandas matplotlib scikit-learn scipy filelock
   ```

## Usage

### Data Preparation
- Place your simulated and experimental data files in the specified paths:
  - Simulated data: `trasmittance_newRecipe_500samples.csv`
  - Experimental training data: `data_exp_train_four_clusterVal.csv`
  - Experimental test data: `data_exp_test_four_clusterVal.csv`

### Running the Model
1. Configure the paths to your data files in the script:
   ```python
   data_dir_sim = 'path/to/simulated_data.csv'
   data_dir_exp_train = 'path/to/experimental_train_data.csv'
   data_dir_exp_test = 'path/to/experimental_test_data.csv'
   ```

2. Set the number of epochs and trials:
   ```python
   epochs = 1  # Number of training epochs per trial
   n_trials = 1  # Number of Optuna optimization trials
   ```

3. Run the script:
   ```bash
   python CGAN_OPTUNA.py
   ```

### Outputs
- **Model Files**:
  - `best_gan_model.h5`: Best generator model.
  - `best_discriminator.h5`: Best discriminator model.
  - `best_params.json`: Best hyperparameters.
  - `best_history.json`: Training history.

- **Visualizations**:
  - Training and validation loss plots.
  - Generated vs. real transmittance curves.
  - Optuna optimization history and parameter importance plots.

## Customization
- **Model Architecture**: Adjust the generator and discriminator architectures in `make_generator_model` and `make_discriminator_model`.
- **Loss Weights**: Modify the weights for different loss components in the `ConditionalGAN` class.
- **Optuna Parameters**: Customize the hyperparameter search space in the `objective` function.

## Example
```python
# Example of loading the best model after training
generator, discriminator, params = load_full_model("best_gan_model.h5", "best_discriminator.h5", "best_params.json")
if generator:
    print("Generator loaded successfully!")
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Disclaimer
This `README` file was generated using LLM and analyzed by the authors.

## Cite
Please, if the present code was used, please cite the manuscript "Physics-Informed Generative Adversarial Networks Applied to Dichroic Filters’ Properties Regression", accordinly.

For questions or issues, please open an issue on the repository.
```
