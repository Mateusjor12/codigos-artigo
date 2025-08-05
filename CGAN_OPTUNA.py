import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
load_model = keras.models.load_model
import optuna
import json
import os
import optuna.visualization as vis_plotly
import optuna.visualization.matplotlib as vis_matplotlib
import matplotlib.pyplot as plt
import signal
import sys
from filelock import FileLock
import shutil
import scipy.interpolate

# Choose CPU and ignore GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BEST_MODEL_PATH = "best_gan_model.h5"
BEST_PARAMS_PATH = "best_params.json"

# Call GPU if available!
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Avoid total memory allocation error
        print("GPU configured correctly!")
    except RuntimeError as e:
        print(e)

# Load data

def load_and_preprocess_data(data_dir_sim, data_dir_exp_train, data_dir_exp_test, test_size=0.2):
    
    df_sim = pd.read_csv(data_dir_sim)

    df_sim.loc[df_sim.index > 103, df_sim.columns[1:]] = 0. # assign zero after the band range

    wavelengths_sim = df_sim.iloc[1:, 0].values.astype(float)  # Extract wavelengths from the first column
    angles_sim = df_sim.columns.values[1:].astype(float)  # Extract angles from column headers
    transmittances_sim = df_sim.iloc[1:, 1:].values.astype(float).T # Transpose
    
    print("Angles shape sim:", angles_sim.shape)
    print("Transmittances shape sim:", transmittances_sim.shape)

    # Normalize 
    angles_norm_sim = (angles_sim - angles_sim.mean()) / angles_sim.std()
    transmittances_norm_sim = transmittances_sim / np.max(transmittances_sim)

    # Split 
    X_train_angle_sim, X_test_angle_sim, y_train_trans_sim, y_test_trans_sim = train_test_split(
        angles_norm_sim, transmittances_norm_sim, test_size=test_size, random_state=42
    )

    train_dataset_sim = tf.data.Dataset.from_tensor_slices(((X_train_angle_sim, tf.random.normal((len(X_train_angle_sim), 100))), y_train_trans_sim)).batch(20)
    test_dataset_sim = tf.data.Dataset.from_tensor_slices(((X_test_angle_sim, tf.random.normal((len(X_test_angle_sim), 100))), y_test_trans_sim)).batch(20)

    #Split of experimental data into training and test sets

    # Training data:
    df_exp_train = pd.read_csv(data_dir_exp_train)

    wavelengths_exp_train = df_exp_train.iloc[1:, 0].values.astype(float)  # Wavelengths
    angles_exp_train = df_exp_train.columns.values[1:].astype(float)  # Angles for training
    transmittances_exp_train = df_exp_train.iloc[1:, 1:].values.astype(float).T  # Transmittance matrix (transposed)

    print("Angles shape exp train:", angles_exp_train.shape)
    print("Transmittances shape exp train:", transmittances_exp_train.shape)

    # Normalization of experimental angles and transmittances 
    angles_norm_exp_train = (angles_exp_train - angles_exp_train.mean()) / angles_exp_train.std()
    transmittances_norm_exp_train = transmittances_exp_train / np.max(transmittances_exp_train)

    # Dataset creation directly (ADDED A SHUFFLE HERE TO VARY REPEATS)
    train_dataset_exp = tf.data.Dataset.from_tensor_slices(
        ((angles_norm_exp_train, tf.random.normal((len(angles_norm_exp_train), 100))), transmittances_norm_exp_train)
    ).shuffle(1000).repeat(5).batch(20)

    # Experimental test data: 
    df_exp_test = pd.read_csv(data_dir_exp_test)

    wavelengths_exp_test = df_exp_test.iloc[1:, 0].values.astype(float)  # Wavelengths
    angles_exp_test = df_exp_test.columns.values[1:].astype(float)  # Angles for test/validation
    transmittances_exp_test = df_exp_test.iloc[1:, 1:].values.astype(float).T  # Transmittance matrix transposed

    print("Angles shape exp test:", angles_exp_test.shape)
    print("Transmittances shape exp test:", transmittances_exp_test.shape)

    # Normalization of experimental angles and transmittances (test)
    angles_norm_exp_test = (angles_exp_test - angles_exp_test.mean()) / angles_exp_test.std()
    transmittances_norm_exp_test = transmittances_exp_test / np.max(transmittances_exp_test)

    # Dataset creation directly for validation
    test_dataset_exp = tf.data.Dataset.from_tensor_slices(
        ((angles_norm_exp_test, tf.random.normal((len(angles_norm_exp_test), 100))), transmittances_norm_exp_test)
    ).repeat(5).batch(20)

    return (
        train_dataset_sim, test_dataset_sim,
        train_dataset_exp, test_dataset_exp,
        angles_sim.mean(), angles_sim.std(),
        angles_exp_train.mean(), angles_exp_train.std(), wavelengths_exp_train
    ) 

    #-------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------

# Function to create the generator model
def make_generator_model(trial=None, gen_num_lstm_layers=1, gen_lstm_units=32, gen_num_dense_layers=5, gen_dense_units=1280, loaded_model=None):
    if loaded_model:
        print("Using loaded model as base for generator.")
        return loaded_model
    
    # Standard code to create the generator
    angle_input = tf.keras.layers.Input(shape=(1,))
    noise_input = tf.keras.layers.Input(shape=(100,))

    # LSTM layers
    if trial:
        num_lstm_layers = trial.suggest_int("gen_num_lstm_layers", 1, 3)
        lstm_units = trial.suggest_int("gen_lstm_units", 16, 80, step=16)
    else:
        num_lstm_layers = gen_num_lstm_layers
        lstm_units = gen_lstm_units

    generator_input_sequence = tf.keras.layers.RepeatVector(100)(noise_input)
    x = generator_input_sequence

    for i in range(num_lstm_layers):
        x = tf.keras.layers.LSTM(units=lstm_units, return_sequences=(i < num_lstm_layers - 1),
                                 name=f"latent_lstm_{i}")(x)

    merged = tf.keras.layers.Concatenate()([angle_input, x])

    # Activation functions
    if trial:
        activation_fn = trial.suggest_categorical("gen_activation", ["relu", "tanh", "selu", "elu"])
    else:
        activation_fn = "relu"

    # Dynamic dense layers
    if trial:
        num_dense_layers = trial.suggest_int("generator_num_dense_layers", 2, 5)
        dense_units = trial.suggest_int("generator_dense_units", 1024, 2048, step=128)
    else:
        num_dense_layers = gen_num_dense_layers
        dense_units = gen_dense_units

    for i in range(num_dense_layers):
        merged = tf.keras.layers.Dense(dense_units, activation=activation_fn, name=f"gen_dense_{i}")(merged)

    # Linear activation function between 0 and 1
    output = tf.keras.layers.Dense(125, activation='sigmoid')(merged)
    return tf.keras.Model(inputs=[angle_input, noise_input], outputs=output)

#-------------------------------------------------------------------------------------------------

def check_disk_space(min_required_gb=1.0):
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    if free_gb < min_required_gb:
        print(f"⚠️ WARNING: Only {free_gb:.2f}GB free! Model saving may fail!")
    return free_gb >= min_required_gb

def save_model_and_params(generator, discriminator, params, model_path, params_path):
    if not check_disk_space():
        print("❌ Insufficient space. Saving canceled.")
        return

    generator.save(model_path)
    discriminator.save("best_discriminator.h5")

    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    print(f"✅ Models saved at '{model_path}' and 'best_discriminator.h5'. Parameters saved at '{params_path}'.")

#-------------------------------------------------------------------------------------------------

def load_full_model(generator_path, discriminator_path, params_path):
    if not os.path.exists(generator_path) or not os.path.exists(params_path):
        print(f"Error: Files '{generator_path}' or '{params_path}' not found.")
        return None, None, None

    try:
        generator = load_model(generator_path, compile=False)  # Avoid automatic recompilation
        discriminator = load_model(discriminator_path, compile=False) if os.path.exists(discriminator_path) else None

        with open(params_path, "r") as f:
            params = json.load(f)

        print("Generator, discriminator loaded successfully!")

        #  MANUAL COMPILATION
        generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                          loss="mse", 
                          metrics=["mae"]) 

        if discriminator:
            discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                                  loss="binary_crossentropy", 
                                  metrics=["accuracy"])  

        return generator, discriminator, params

    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None


# define the discriminator model

def make_discriminator_model(trial=None, disc_num_dense_layers=4, disc_dense_units=512):
    angle_input = tf.keras.layers.Input(shape=(1,))
    transmittance_input = tf.keras.layers.Input(shape=(125,))
    merged = tf.keras.layers.Concatenate()([angle_input, transmittance_input])

    # Activation function suggestions
    if trial:
        activation_fn = trial.suggest_categorical("disc_activation", ["relu", "tanh", "selu", "elu"])
    else:
        activation_fn = "tanh"

    # Dynamic dense layers
    if trial:
        num_dense_layers = trial.suggest_int("discriminator_num_dense_layers", 2, 4)
        dense_units = trial.suggest_int("discriminator_dense_units", 256, 1024, step=128)
    else:
        num_dense_layers = disc_num_dense_layers
        dense_units = disc_dense_units

    x = merged
    for i in range(num_dense_layers):
        x = tf.keras.layers.Dense(dense_units, activation=activation_fn, name=f"disc_dense_{i}")(x)

    # Last layer: sigmoid activation function
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=[angle_input, transmittance_input], outputs=output)

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

# Losses for CGAN: 
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total = real_loss + fake_loss
    return total

def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss

def mse_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std):

    normalized_60 = (60 - angles_mean) / angles_std

    real_transmittances = tf.cast(real_transmittances, dtype=tf.float32)
    generated_transmittances = tf.cast(generated_transmittances, dtype=tf.float32)
    angles = tf.cast(angles, dtype=tf.float32)

    loss = tf.reduce_mean(tf.square(real_transmittances - generated_transmittances))
    
    penalty_mask = tf.where(angles > normalized_60, 1.3, 1.0)  
    loss = tf.reduce_mean(loss * penalty_mask)

    return loss

def cosine_similarity_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std):
 
    normalized_60 = (60 - angles_mean) / angles_std

    real_transmittances = tf.cast(real_transmittances, dtype=tf.float32)
    generated_transmittances = tf.cast(generated_transmittances, dtype=tf.float32)
    angles = tf.cast(angles, dtype=tf.float32)

    real_normalized = tf.math.l2_normalize(real_transmittances, axis=1)
    generated_normalized = tf.math.l2_normalize(generated_transmittances, axis=1)

    loss = 1 - tf.reduce_mean(tf.reduce_sum(real_normalized * generated_normalized, axis=1))
    
    penalty_mask = tf.where(angles > normalized_60, 1.3, 1.0)  
    loss = tf.reduce_mean(loss * penalty_mask)

    return loss

def interpolate_to_reference(curves, source_wavelengths, target_wavelengths):
    """
    Interpolates each curve in 'curves' from the domain 'source_wavelengths' to 'target_wavelengths'.
    Returns a matrix with the same shape (batch_size, len(target_wavelengths)).
    """
    interpolated_curves = []

    for i in range(curves.shape[0]):

        interp_fn = scipy.interpolate.interp1d(
            source_wavelengths[i], curves[i],
            kind='linear', fill_value='extrapolate'
        )
        interpolated = interp_fn(target_wavelengths[i])
        interpolated_curves.append(interpolated)

    return tf.convert_to_tensor(interpolated_curves, dtype=tf.float32)

def find_10_percent_max_wavelengths(x, y):
    """
    Finds the initial and final wavelengths where transmittance is above 10% of the maximum value.
    """
    epsilon = 1e-6  # Small value to avoid division by zero
    max_vals = tf.reduce_max(y, axis=1, keepdims=True) + epsilon
    threshold = 0.1 * max_vals

    mask_above_threshold = tf.cast(y > threshold, tf.float32)

    # Indices of the first and last point above the threshold
    start_idx = tf.argmax(mask_above_threshold, axis=1, output_type=tf.int32)
    end_idx = tf.shape(y)[1] - 1 - tf.argmax(tf.reverse(mask_above_threshold, axis=[1]), axis=1, output_type=tf.int32)

    # Gets the maximum valid index in the vector `x`
    max_index = tf.shape(x)[-1] - 1  # Last valid index

    #  Fix possible out-of-bounds indices
    start_idx = tf.clip_by_value(start_idx, 0, max_index)
    end_idx = tf.clip_by_value(end_idx, 0, max_index)

    # DEBUG: Inspect values before using `tf.gather()`**
    if tf.executing_eagerly():
        print(f"DEBUG: x shape = {x.shape}")
        print(f"DEBUG: max_index = {max_index.numpy()}")
        print(f"DEBUG: start_idx = {start_idx.numpy()}, end_idx = {end_idx.numpy()}")

    x = tf.reshape(x, [-1])  # Transform into 1D vector to avoid shape error

    # Ensure indices are valid before using tf.gather()
    valid_start_idx = tf.where(start_idx <= max_index, start_idx, max_index)
    valid_end_idx = tf.where(end_idx <= max_index, end_idx, max_index)

    # Returns the corrected initial and final wavelengths
    return tf.gather(x, valid_start_idx), tf.gather(x, valid_end_idx)

def band_loss(real_transmittances, generated_transmittances, wavelengths):
    """
    Calculates the band loss between real and generated transmittances based on the 10% points of maximum transmittance.
    """
    # Convert to float32
    real_transmittances = tf.cast(real_transmittances, dtype=tf.float32)
    generated_transmittances = tf.cast(generated_transmittances, dtype=tf.float32)
    wavelengths = tf.cast(wavelengths, dtype=tf.float32)

    batch_size = tf.shape(real_transmittances)[0]
    wavelengths_real = tf.tile(tf.expand_dims(wavelengths, axis=0), [batch_size, 1])
    wavelengths_gen = tf.tile(tf.expand_dims(wavelengths, axis=0), [batch_size, 1]) 

    # If different, apply interpolation
    if not tf.reduce_all(tf.equal(wavelengths_real, wavelengths_gen)):
        generated_transmittances = interpolate_to_reference(generated_transmittances, wavelengths_gen, wavelengths_real)

    start_true, end_true = find_10_percent_max_wavelengths(wavelengths_real, real_transmittances)
    start_pred, end_pred = find_10_percent_max_wavelengths(wavelengths_real, generated_transmittances)

    # 🔍 **DEBUG: Print extracted values**
    if tf.executing_eagerly():
        print(f"Start True Wavelengths: {start_true.numpy()}")
        print(f"Start Predicted Wavelengths: {start_pred.numpy()}")
        print(f"Diff Start: {np.abs(start_true.numpy() - start_pred.numpy())}")

        print(f"End True Wavelengths: {end_true.numpy()}")
        print(f"End Predicted Wavelengths: {end_pred.numpy()}")
        print(f"Diff End: {np.abs(end_true.numpy() - end_pred.numpy())}")

    # Normalize loss with the total wavelength range
    total_wavelength_range = tf.reduce_max(wavelengths) - tf.reduce_min(wavelengths) + 1e-6  # Avoid division by zero

    # Calculate MSE between start and end values
    start_loss = tf.reduce_mean(tf.square(start_true - start_pred) / total_wavelength_range)
    end_loss = tf.reduce_mean(tf.square(end_true - end_pred) / total_wavelength_range)

    return start_loss + end_loss


def derivative_loss(real_transmittances, generated_transmittances, wavelengths, penalty_weight):
    
    #Calculates the loss based on the difference of the derivatives of the transmittance curves.
    
    # Convert to numpy float32 before creating the tensor
    real_transmittances = tf.cast(real_transmittances, dtype=tf.float32)
    generated_transmittances = tf.cast(generated_transmittances, dtype=tf.float32)
    wavelengths = tf.cast(wavelengths, dtype=tf.float32)

    batch_size = tf.shape(real_transmittances)[0]
    wavelengths_real = tf.tile(tf.expand_dims(wavelengths, axis=0), [batch_size, 1])
    wavelengths_gen = tf.tile(tf.expand_dims(wavelengths, axis=0), [batch_size, 1])  # If different, will be handled

    # Align domains by interpolation if different
    if not tf.reduce_all(tf.equal(wavelengths_real, wavelengths_gen)):
        generated_transmittances = interpolate_to_reference(generated_transmittances, wavelengths_gen, wavelengths_real)
        
    # Ensure tensors have at least 2 dimensions
    if tf.rank(real_transmittances) == 1:
        real_transmittances = tf.expand_dims(real_transmittances, axis=0)

    if len(generated_transmittances.shape) == 1:
        generated_transmittances = tf.expand_dims(generated_transmittances, axis=0)

    if len(wavelengths.shape) == 1:
        wavelengths = tf.expand_dims(wavelengths, axis=0)

    # Avoid out-of-bounds slicing
    if real_transmittances.shape[1] < 2:
        return tf.constant(0.0, dtype=tf.float32)

    dT_real = (real_transmittances[:, 1:] - real_transmittances[:, :-1]) / (wavelengths[:, 1:] - wavelengths[:, :-1])
    dT_gen = (generated_transmittances[:, 1:] - generated_transmittances[:, :-1]) / (wavelengths[:, 1:] - wavelengths[:, :-1])

    if dT_real.shape[1] < 2:
        return tf.constant(0.0, dtype=tf.float32)

    ddT_real = (dT_real[:, 1:] - dT_real[:, :-1]) / (wavelengths[:, 2:] - wavelengths[:, :-2])
    ddT_gen = (dT_gen[:, 1:] - dT_gen[:, :-1]) / (wavelengths[:, 2:] - wavelengths[:, :-2])

    first_derivative_loss = tf.reduce_mean(tf.square(dT_real - dT_gen))
    second_derivative_penalty = tf.reduce_mean(tf.square(ddT_real - ddT_gen))
  
    return first_derivative_loss + penalty_weight * second_derivative_penalty

def huber_loss(y_true, y_pred, angles, angles_mean, angles_std, delta=1.0):
    
    #Huber Loss: a combination of MSE and MAE that avoids extreme penalties for outliers.
    
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    angles = tf.cast(angles, dtype=tf.float32)

    error = y_true - y_pred
    is_small_error = tf.abs(error) < delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)

    loss = tf.where(is_small_error, squared_loss, linear_loss)

    normalized_60 = (60 - angles_mean) / angles_std
    penalty_mask = tf.where(angles > normalized_60, 1.3, 1.0)

    return tf.reduce_mean(loss * tf.expand_dims(penalty_mask, axis=-1))

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

# Define the ConditionalGAN model
class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, gweight=1.0, mse_weight=2, cosine_weight=3, 
                 experimental_mse_weight=40, experimental_cosine_weight=38, 
                 derivative_weight=45, huber_weight=18, 
                 bands_weight=25, penalty_weight=10, experimental_huber_weight=15):  
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.mse_weight = mse_weight
        self.gweight = gweight 
        self.cosine_weight = cosine_weight
        self.experimental_mse_weight = experimental_mse_weight
        self.experimental_cosine_weight = experimental_cosine_weight
        self.derivative_weight = derivative_weight
        self.huber_weight = huber_weight
        self.bands_weight = bands_weight
        self.penalty_weight = penalty_weight
        self.experimental_huber_weight = experimental_huber_weight


    def compile(self, d_optimizer, g_optimizer):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
    
    def train_step(self, angles, noise, real_transmittances, experimental_angles, 
                   experimental_transmittances, noise_exp, wavelengths, angles_mean, 
                   angles_std, angles_exp_mean, angles_exp_std):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate transmittance curves
            generated_transmittances = self.generator([angles, noise], training=True)
            experimental_gen_transmittances = self.generator([experimental_angles, noise_exp], training=True)

            # Losses based on curves
            deriv_loss_value = derivative_loss(experimental_transmittances, experimental_gen_transmittances, wavelengths, self.penalty_weight)
            band_loss_value = band_loss(experimental_transmittances, experimental_gen_transmittances, wavelengths)

            # Standard CGAN loss calculations
            real_output = self.discriminator([angles, real_transmittances], training=True)
            fake_output = self.discriminator([angles, generated_transmittances], training=True)
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            mse_loss_value = mse_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)
            cosine_loss_value = cosine_similarity_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)

            experimental_mse_loss = mse_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_exp_mean, angles_exp_std)
            experimental_cosine_loss = cosine_similarity_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_exp_mean, angles_exp_std)

            huber_loss_value = huber_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)
            experimental_huber_loss = huber_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_exp_mean, angles_exp_std)

            # Total generator loss
            total_gen_loss = tf.cast(self.gweight * gen_loss, dtype=tf.float32) + \
                tf.cast(self.mse_weight * mse_loss_value, dtype=tf.float32) + \
                tf.cast(self.cosine_weight * cosine_loss_value, dtype=tf.float32) + \
                tf.cast(self.experimental_mse_weight * experimental_mse_loss, dtype=tf.float32) + \
                tf.cast(self.experimental_cosine_weight * experimental_cosine_loss, dtype=tf.float32) + \
                tf.cast(self.derivative_weight * deriv_loss_value, dtype=tf.float32) + \
                tf.cast(self.huber_weight * huber_loss_value, dtype=tf.float32) + \
                tf.cast(self.bands_weight * band_loss_value, dtype=tf.float32) + tf.cast(self.experimental_huber_weight * experimental_huber_loss, dtype=tf.float32)

        # Update weights
        gradients_of_generator = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {
            'd_loss': disc_loss,
            'g_loss': gen_loss,
            'mse_loss': mse_loss_value,
            'cosine_loss': cosine_loss_value,
            'experimental_mse_loss': experimental_mse_loss,
            'experimental_cosine_loss': experimental_cosine_loss,
            'derivative_loss': deriv_loss_value,
            'train_huber_loss': huber_loss_value,
            'experimental_huber_loss': experimental_huber_loss,
            'band_loss': band_loss_value
        }

    def test_step(self, angles, noise, real_transmittances, experimental_angles, 
                  experimental_transmittances, noise_exp, wavelengths, angles_mean, 
                  angles_std, angles_exp_mean, angles_exp_std):
        
        # Generate transmittance curves
        generated_transmittances = self.generator([angles, noise], training=False)
        experimental_gen_transmittances = self.generator([experimental_angles, noise_exp], training=False)

        # Standard CGAN loss calculations
        real_output = self.discriminator([angles, real_transmittances], training=False)
        fake_output = self.discriminator([angles, generated_transmittances], training=False)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        mse_loss_value = mse_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)
        cosine_loss_value = cosine_similarity_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)

        experimental_mse_loss = mse_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_exp_mean, angles_exp_std)
        experimental_cosine_loss = cosine_similarity_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_exp_mean, angles_exp_std)

        # Additional losses for evaluation, compatible with train_step()
        deriv_loss_value = derivative_loss(experimental_transmittances, experimental_gen_transmittances, wavelengths, self.penalty_weight)
        band_loss_value = band_loss(real_transmittances, generated_transmittances, wavelengths)

        """bootstrap_curves = generate_bootstrap_curves(tf.identity(experimental_transmittances), n_bootstrap=100)
        bootstrap_loss_value = tf.reduce_mean(tf.reduce_mean(tf.square(bootstrap_curves - tf.expand_dims(experimental_transmittances, axis=1)), axis=2))"""

        huber_loss_value = huber_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)
        experimental_huber_loss = huber_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_exp_mean, angles_exp_std)

        return {
            'd_loss': disc_loss,
            'g_loss': gen_loss,
            'mse_loss': mse_loss_value,
            'cosine_loss': cosine_loss_value,
            'experimental_mse_loss': experimental_mse_loss,
            'experimental_cosine_loss': experimental_cosine_loss,
            'derivative_loss': deriv_loss_value,
            'test_huber_loss': huber_loss_value,  
            'experimental_huber_loss': experimental_huber_loss,
            'band_loss': band_loss_value
        }

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

# Visualization and denormalization functions

def denormalize(data, mean, std):
    return data * std + mean

def visualize_results(gan, test_dataset, exp_dataset, angles_mean, \
                      angles_std, angles_exp_mean, angles_exp_std, history,
                      num_samples=10):

    # Getting a batch of test samples
    try:
        test_batch = next(iter(test_dataset.take(1)))
        exp_batch = next(iter(exp_dataset.take(1)))
    except Exception as e:
        print(f"Error extracting batch from datasets: {e}")
        return

    (angles, noise), real_transmittances = test_batch
    (angles_exp, noise_exp), exp_transmittances = exp_batch

    # Generate predictions for test and experimental datasets
    generated_transmittances = gan.generator.predict([angles, noise])
    generated_transmittances_exp = gan.generator.predict([angles_exp, noise_exp])

    # Subplot layout definition
    ncols = 4
    nrows = (num_samples + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))

    for i in range(num_samples):
        row = i // ncols
        col = i % ncols

        denorm_real_angle = denormalize(angles[i].numpy(), angles_mean, angles_std)
        denorm_real_transmittance = real_transmittances[i].numpy()
        denorm_gen_transmittance = generated_transmittances[i]

        axes[row, col].plot(denorm_real_transmittance, label=f'Real (Angle: {denorm_real_angle:.2f})')
        axes[row, col].plot(denorm_gen_transmittance, label='Generated')
        axes[row, col].set_xlabel('Wavelength Point')
        axes[row, col].set_ylabel('Transmittance')
        axes[row, col].set_title(f'Sample {i+1}')
        axes[row, col].legend()

    # Remove empty subplots
    for j in range(num_samples, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    plt.suptitle('Model vs. Generated Transmittance Curves', fontsize=16)
    plt.savefig('./Model_vs_Generated_Transmittance_Curves.png')

# ---------- Plot Experimental Data ----------

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))

    for i in range(num_samples):
        row = i // ncols
        col = i % ncols

        # Denormalization of experimental angles
        denorm_exp_angle = denormalize(angles_exp[i].numpy(), angles_exp_mean, angles_exp_std)
        denorm_exp_transmittance = exp_transmittances[i].numpy()
        denorm_gen_transmittance_exp = generated_transmittances_exp[i]
        axes[row, col].plot(denorm_exp_transmittance, label=f'Real Exp (Angle: {denorm_exp_angle:.2f})')
        axes[row, col].plot(denorm_gen_transmittance_exp, label='Generated Exp')
        axes[row, col].set_xlabel('Wavelength Point')
        axes[row, col].set_ylabel('Transmittance')
        axes[row, col].set_title(f'Exp Sample {i+1}')
        axes[row, col].legend()

    for j in range(num_samples, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    plt.suptitle('Real vs. Generated Experimental Transmittance Curves', fontsize=16)
    plt.savefig('./Real_vs_Generated_Experimental_Transmittance_Curves.png')

    # ---------- Plotting History ----------

    if history:
        plt.figure(figsize=(12, 8))
        plt.plot(history['d_loss'], label='Discriminator Loss')
        plt.plot(history['g_loss'], label='Generator Loss')
        plt.plot(history['mse_loss'], label='MSE Loss')
        plt.plot(history['cosine_loss'], label='Cosine Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.savefig('./Training_Losses.png')

        plt.figure(figsize=(12, 8))
        plt.plot(history['mse_loss'], label='Train MSE Loss')
        plt.plot(history['cosine_loss'], label='Train Cosine Loss')
        plt.plot(history['val_mse_loss'], label='Validation MSE Loss')
        plt.plot(history['val_cosine_loss'], label='Validation Cosine Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Losses')
        plt.legend()
        plt.savefig('./Train_Validation_Losses.png')

        plt.figure(figsize=(12, 8))
        plt.plot(history['experimental_mse_loss'], label='Train Experimental MSE Loss')
        plt.plot(history['experimental_cosine_loss'], label='Train Experimental Cosine Loss')
        plt.plot(history['val_experimental_mse_loss'], label='Val Experimental MSE Loss')
        plt.plot(history['val_experimental_cosine_loss'], label='Val Experimental Cosine Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Experimental Train and Validation Losses')
        plt.legend()
        plt.savefig('./Experimental_Train_Validation_Losses.png')
    else:
        print("⚠️ Warning: No training history found. Loss plots were not generated.")

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

# Visualize

def save_fig(fig, filename, save_dir):
    # Helper function to properly save matplotlib plots
    if fig is None:
        print(f"⚠️ No plot generated for {filename}. Skipping...")
        return

    if isinstance(fig, plt.Axes):
        fig = fig.get_figure()

    if isinstance(fig, plt.Figure):
        fig.savefig(os.path.join(save_dir, f"{filename}.png"))
        plt.close(fig)
    else:
        print(f"⚠️ Unexpected format for {filename}. Skipping...")

def save_plotly(fig, filename, save_dir):
     # Helper function to properly save Plotly plots.
    if fig is None:
        print(f"⚠️ No plot generated for {filename}. Skipping...")
        return
    fig.write_html(os.path.join(save_dir, f"{filename}.html"))

def visualize_optuna_results(study):
    
    #Generates and saves all available visualizations for Optuna optimization analysis.
    #Plots will be saved in both PNG (Matplotlib) and HTML (Plotly) for later analysis.
   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "optuna_results")
    os.makedirs(save_dir, exist_ok=True)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) == 0:
        print("⚠️ No completed trial available in the study. Nothing to visualize.")
        return

    print(f"📊 Generating and saving Optuna plots in '{save_dir}'...")

    # 🛠️ **Check which parameters actually exist in the study**
    existing_params = list(study.best_params.keys())
    print(f"✅ Existing parameters in the study: {existing_params}")

    # Choose safe parameters for the plots
    safe_params = ["generator_num_dense_layers", "generator_dense_units"]
    selected_params = [param for param in safe_params if param in existing_params]

    try:
        # Matplotlib plots (static)
        save_fig(vis_matplotlib.plot_optimization_history(study), "optimization_history", save_dir)
        save_fig(vis_matplotlib.plot_intermediate_values(study), "intermediate_values", save_dir)
        save_fig(vis_matplotlib.plot_parallel_coordinate(study), "parallel_coordinate", save_dir)

        if selected_params:
            save_fig(vis_matplotlib.plot_parallel_coordinate(study, params=selected_params), "parallel_coordinate_selected", save_dir)
            save_fig(vis_matplotlib.plot_contour(study, params=selected_params), "contour_plot_selected", save_dir)
            save_fig(vis_matplotlib.plot_slice(study, params=selected_params), "slice_plot_selected", save_dir)

        save_fig(vis_matplotlib.plot_contour(study), "contour_plot", save_dir)
        save_fig(vis_matplotlib.plot_slice(study), "slice_plot", save_dir)
        save_fig(vis_matplotlib.plot_param_importances(study), "param_importances", save_dir)
        save_fig(vis_matplotlib.plot_edf(study), "edf_plot", save_dir)
        save_fig(vis_matplotlib.plot_rank(study), "rank_plot", save_dir)
        save_fig(vis_matplotlib.plot_timeline(study), "timeline_plot", save_dir)

    except Exception as e:
        print(f"⚠️ Error generating Matplotlib plots: {e}")

    try:
        # Plotly plots (interactive)
        save_plotly(vis_plotly.plot_optimization_history(study), "optimization_history_interactive", save_dir)
        save_plotly(vis_plotly.plot_intermediate_values(study), "intermediate_values_interactive", save_dir)
        save_plotly(vis_plotly.plot_parallel_coordinate(study), "parallel_coordinate_interactive", save_dir)

        if selected_params:
            save_plotly(vis_plotly.plot_parallel_coordinate(study, params=selected_params), "parallel_coordinate_selected_interactive", save_dir)
            save_plotly(vis_plotly.plot_contour(study, params=selected_params), "contour_plot_selected_interactive", save_dir)
            save_plotly(vis_plotly.plot_slice(study, params=selected_params), "slice_plot_selected_interactive", save_dir)

        save_plotly(vis_plotly.plot_contour(study), "contour_plot_interactive", save_dir)
        save_plotly(vis_plotly.plot_slice(study), "slice_plot_interactive", save_dir)
        save_plotly(vis_plotly.plot_param_importances(study), "param_importances_interactive", save_dir)
        save_plotly(vis_plotly.plot_edf(study), "edf_plot_interactive", save_dir)
        save_plotly(vis_plotly.plot_rank(study), "rank_plot_interactive", save_dir)
        save_plotly(vis_plotly.plot_timeline(study), "timeline_plot_interactive", save_dir)

    except Exception as e:
        print(f"⚠️ Error generating Plotly plots: {e}")

    print(f"✅ All Optuna visualizations have been saved in '{save_dir}' successfully!")

    
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

# OPTUNA

# Training Loop

def train(mse_weight, 
          cosine_weight,
          experimental_mse_weight, 
          experimental_cosine_weight,
          derivative_weight,
          huber_weight,
          experimental_huber_weight,
          bands_weight,
          penalty_weight,
          angles_mean,
          angles_std,
          angles_exp_mean,
          angles_exp_std,
          glr,
          dlr,
          epochs,
          train_dataset_sim,  
          test_dataset_sim, 
          train_dataset_exp, 
          test_dataset_exp,
          gweight,
          wavelengths,
          trial=None,
          gen_num_lstm_layers=1, gen_lstm_units=64, 
          gen_num_dense_layers=2, gen_dense_units=512,
          disc_num_dense_layers=1, disc_dense_units=512,
          gen_activation="relu", disc_activation="relu",
          loaded_model=None): 

    # Pass the loaded model to the generator
    generator = make_generator_model(trial, gen_num_lstm_layers, gen_lstm_units, gen_num_dense_layers, gen_dense_units, loaded_model=loaded_model)
    discriminator = make_discriminator_model(trial, disc_num_dense_layers, disc_dense_units)

    gan = ConditionalGAN(discriminator, generator, gweight=gweight, mse_weight=mse_weight, cosine_weight=cosine_weight, 
                         experimental_mse_weight=experimental_mse_weight, experimental_cosine_weight=experimental_cosine_weight,
                         derivative_weight=derivative_weight, huber_weight=huber_weight, experimental_huber_weight=experimental_huber_weight, bands_weight=bands_weight, penalty_weight=penalty_weight)

    d_optimizer = tf.keras.optimizers.Adam(glr)
    g_optimizer = tf.keras.optimizers.Adam(dlr)
    gan.compile(d_optimizer, g_optimizer)

    history = {'d_loss': [], 'g_loss': [], 'mse_loss': [], 'cosine_loss': [], 'experimental_mse_loss' : [], 'experimental_cosine_loss': [], 
            'val_d_loss': [], 'val_g_loss': [], 'val_mse_loss': [], 'val_cosine_loss': [], 'val_experimental_mse_loss' : [], 'val_experimental_cosine_loss': []}

    for epoch in range(epochs):
        # Training Step
        for (batch_sim, batch_exp) in zip(train_dataset_sim, train_dataset_exp):
            (angles_sim, noise_sim), real_transmittances_sim = batch_sim
            (angles_exp, noise_exp), experimental_transmittances = batch_exp
            train_loss = gan.train_step(
                angles_sim,
                noise_sim,
                real_transmittances_sim,
                angles_exp,
                experimental_transmittances,
                noise_exp,
                wavelengths,  
                angles_mean,       
                angles_std,       
                angles_exp_mean,   
                angles_exp_std     
            )

        # Testing Step
        for (batch_test_sim, batch_test_exp) in zip(test_dataset_sim, test_dataset_exp):
            (angles_sim, noise_sim), real_transmittances_sim = batch_test_sim
            (angles_exp, noise_exp), experimental_transmittances = batch_test_exp
            test_loss = gan.test_step(
                angles_sim,
                noise_sim,
                real_transmittances_sim,
                angles_exp,
                experimental_transmittances,
                noise_exp,
                wavelengths,  
                angles_mean,       
                angles_std,        
                angles_exp_mean,   
                angles_exp_std     

            )

        # Update loss history after each epoch
        history['d_loss'].append(train_loss['d_loss'].numpy())
        history['g_loss'].append(train_loss['g_loss'].numpy())
        history['mse_loss'].append(train_loss['mse_loss'].numpy())
        history['cosine_loss'].append(train_loss['cosine_loss'].numpy())
        history['experimental_mse_loss'].append(train_loss['experimental_mse_loss'].numpy())
        history['experimental_cosine_loss'].append(train_loss['experimental_cosine_loss'].numpy())

        history['val_d_loss'].append(test_loss['d_loss'].numpy())
        history['val_g_loss'].append(test_loss['g_loss'].numpy())
        history['val_mse_loss'].append(test_loss['mse_loss'].numpy())
        history['val_cosine_loss'].append(test_loss['cosine_loss'].numpy())
        history['val_experimental_mse_loss'].append(test_loss['experimental_mse_loss'].numpy())
        history['val_experimental_cosine_loss'].append(test_loss['experimental_cosine_loss'].numpy())

        print(f'Epoch {epoch+1}, d_loss: {train_loss["d_loss"].numpy():.4f}, g_loss: {train_loss["g_loss"].numpy():.4f}, mse_loss: {train_loss["mse_loss"].numpy():.4f}, cosine_loss: {train_loss["cosine_loss"].numpy():.4f}, exp_mse: {train_loss["experimental_mse_loss"].numpy():.4f}, exp_cos: {train_loss["experimental_cosine_loss"].numpy():.4f}')
        print(f'Epoch {epoch+1}, val_d_loss: {test_loss["d_loss"].numpy():.4f}, val_g_loss: {test_loss["g_loss"].numpy():.4f}, val_mse_loss: {test_loss["mse_loss"].numpy():.4f}, val_cosine_loss: {test_loss["cosine_loss"].numpy():.4f}, val_exp_mse: {test_loss["experimental_mse_loss"].numpy():.4f}, val_exp_cos: {test_loss["experimental_cosine_loss"].numpy():.4f}')

        #POSSIBLE CHANGE HERE?
        if history['experimental_mse_loss']:
            current_value = 0.9 * (history['experimental_mse_loss'][-1] + history['experimental_cosine_loss'][-1]) + \
                            0.1 * (history['mse_loss'][-1] + history['cosine_loss'][-1])
        else:
            current_value = float('inf')  # Avoid error if no values


        if trial:
            trial.report(current_value, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # Return the final experimental MSE and cosine losses
    loss_final = {'final_experimental_mse_loss': history['experimental_mse_loss'][-1],
                'final_experimental_cosine_loss': history['experimental_cosine_loss'][-1],
                'final_mse_loss': history['mse_loss'][-1],
                'final_cosine_loss': history['cosine_loss'][-1],
                'final_val_d_loss': history['val_d_loss'][-1],
                'final_val_g_loss': history['val_g_loss'][-1],
                'final_d_loss' : history['d_loss'][-1],
                'final_g_loss' : history['g_loss'][-1]}
    
    # Saved history path
    history_path = "best_history.json"

    # Check if there is a valid training history before saving
    if history and isinstance(history, dict):
        # Convert float before saving
        history = {key: [float(val) for val in values] for key, values in history.items()}

        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        print("✅ Training history saved successfully!")
    else:
        print("⚠️ Training history is empty or invalid. Nothing was saved.")


    return loss_final, history, gan, current_value

#----------------------------------------------------------------

# Function to ensure proper shutdown
def graceful_exit(signum, frame):
    print("\n🚨 Interruption detected! Saving progress and closing properly...")
    study.trials_dataframe().to_csv("optuna_backup.csv", index=False)
    sys.exit(0)

# Capture interruption (CTRL+C) and termination signals
signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

#----------------------------------------------------------------

# Saving progress in the SQLite database optuna_cgan_study!
STORAGE_NAME = "sqlite:///optuna_cgan_study.db"
BEST_MODEL_PATH = "best_gan_model.h5"
BEST_PARAMS_PATH = "best_params.json"

LOCK_PATH = "optuna_db.lock"

# Create the study globally before the objective function
study = optuna.create_study(
    direction='minimize',
    storage=STORAGE_NAME,
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(multivariate=True),  # Optimized TPE
    pruner=optuna.pruners.HyperbandPruner(min_resource=10, reduction_factor=2)
)

def objective(trial):
    epochs = 1
    loaded_model = load_model(BEST_MODEL_PATH) if trial.number == 0 and os.path.exists(BEST_MODEL_PATH) else None

    # Check if there are saved hyperparameters and load them
    best_params = {}
    if os.path.exists(BEST_PARAMS_PATH):
        try:
            with open(BEST_PARAMS_PATH, "r") as f:
                best_params = json.load(f)
            print("Hyperparameters loaded from previous study:", best_params)
        except Exception as e:
            print(f"Error loading best_params: {e}")
            best_params = {}  # If failed, initialize as empty dict

    # Define hyperparameters with previous values or Optuna suggestion
    gen_num_lstm_layers = trial.suggest_int("gen_num_lstm_layers", 1, 3)
    gen_lstm_units = trial.suggest_int("gen_lstm_units", 32, 80, step=16)
    gen_num_dense_layers = trial.suggest_int("generator_num_dense_layers", 2, 5)
    gen_dense_units = trial.suggest_int("generator_dense_units", 1024, 2048, step=128)
    gen_activation = trial.suggest_categorical("gen_activation", ["relu", "tanh", "selu", "elu"])

    disc_num_dense_layers = trial.suggest_int("discriminator_num_dense_layers", 2, 4)
    disc_dense_units = trial.suggest_int("discriminator_dense_units", 256, 1024, step=128)
    disc_activation = trial.suggest_categorical("disc_activation", ["relu", "tanh", "selu", "elu"])

    mse_weight = trial.suggest_float("mse_weight", 0.1, 30)
    cosine_weight = trial.suggest_float("cosine_weight", 0.1, 30)
    experimental_mse_weight = trial.suggest_float("experimental_mse_weight", 1.0, 50.0)
    experimental_cosine_weight = trial.suggest_float("experimental_cosine_weight", 1.0, 50.0)
    gweight = trial.suggest_float("generator_weight", 1.0, 50.0)

    derivative_weight = trial.suggest_float("derivative_weight", 1.0, 45.0)
    huber_weight = trial.suggest_float("huber_weight", 1.0, 50.0)
    experimental_huber_weight = trial.suggest_float("experimental_huber_weight", 1.0, 50.0) 

    bands_weight = trial.suggest_float("bands_weight", 1.0, 50.0)
    penalty_weight = trial.suggest_float("penalty_weight", 1.0, 50.0)

    glr = 1e-4  # generator learning rate (fixed value)
    dlr = 1e-5  # discriminator learning rate (fixed value)

    # Training with suggested hyperparameters
    loss_final, _, gan, current_value = train(
        mse_weight, 
        cosine_weight,
        experimental_mse_weight, 
        experimental_cosine_weight,
        derivative_weight,
        huber_weight,
        experimental_huber_weight,
        bands_weight,
        penalty_weight,
        angles_mean,
        angles_std,
        angles_exp_mean,
        angles_exp_std,
        glr,
        dlr,
        epochs,
        train_dataset_sim, 
        test_dataset_sim,
        train_dataset_exp, 
        test_dataset_exp,
        gweight=gweight,
        wavelengths=wavelengths_exp,
        trial=trial,
        gen_num_lstm_layers=gen_num_lstm_layers,
        gen_lstm_units=gen_lstm_units,
        gen_num_dense_layers=gen_num_dense_layers,
        gen_dense_units=gen_dense_units,
        disc_num_dense_layers=disc_num_dense_layers,
        disc_dense_units=disc_dense_units,
        gen_activation=gen_activation,
        disc_activation=disc_activation,
        loaded_model=loaded_model
        )

    # Check the best value of the study safely
    best_value = study.best_value if study.best_trials else float('inf')

    if current_value <= best_value * 1.1 or trial.number % 5 == 0:
        print("New best model found! Saving...")

        best_params = {}
        if os.path.exists(BEST_PARAMS_PATH):
            try:
                with open(BEST_PARAMS_PATH, "r") as f:
                    best_params = json.load(f)
                print("Hyperparameters loaded from previous study:", best_params)
            except Exception as e:
                print(f"Error loading best_params: {e}")
                best_params = {}  # If failed, initialize as empty dict
            
        save_model_and_params(gan.generator, gan.discriminator, {
            "mse_weight": mse_weight,
            "cosine_weight": cosine_weight,
            "experimental_mse_weight": experimental_mse_weight,
            "experimental_cosine_weight": experimental_cosine_weight,
            "derivative_weight": derivative_weight,
            "huber_weight": huber_weight,
            "experimental_huber_weight": experimental_huber_weight,
            "bands_weight": bands_weight,
            "penalty_weight": penalty_weight,
            "generator_weight": gweight,
            "generator_num_dense_layers": gen_num_dense_layers,
            "generator_dense_units": gen_dense_units,
            "generator_activation": gen_activation,
            "discriminator_num_dense_layers": disc_num_dense_layers,
            "discriminator_dense_units": disc_dense_units,
            "discriminator_activation": disc_activation,
            "gen_num_lstm_layers": gen_num_lstm_layers,
            "gen_lstm_units": gen_lstm_units,
            "generator_lr": 1e-4, 
            "discriminator_lr": 1e-5
        }, BEST_MODEL_PATH, BEST_PARAMS_PATH)

        trial.set_user_attr("model_path", BEST_MODEL_PATH)
    else:
        print("Current model is not better than the previously saved one.")

    if trial.number % 20 == 0:  # Save backup every 20 trials
        backup_path = f"optuna_cgan_study_backup_{trial.number}.db"
        shutil.copy("optuna_cgan_study.db", backup_path)
        print(f"🛠️ Backup created: {backup_path}")

    return current_value

# RUNNING
data_dir_sim = 'C:/Users/Usuario/Desktop/IA em dados Óticos de materiais/trasmittance_newRecipe_500samples.csv' 
data_dir_exp_train = 'C:/Users/Usuario/Desktop/IA em dados Óticos de materiais/data_exp_train_four_clusterVal.csv'
data_dir_exp_test = 'C:/Users/Usuario/Desktop/IA em dados Óticos de materiais/data_exp_test_four_clusterVal.csv' 
train_dataset_sim, test_dataset_sim, train_dataset_exp, test_dataset_exp, angles_mean, angles_std, angles_exp_mean, angles_exp_std, wavelengths_exp = load_and_preprocess_data(data_dir_sim, data_dir_exp_train, data_dir_exp_test)

epochs = 1 # Set the number of epochs for training
n_trials = 1  # Set the number of trials for Optuna

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Run the OPTUNA optimization
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# Run the study with `n_trials` experiments

try:
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
except Exception as e:
    print(f"Error during optimization: {e}")

# Print the best parameters
best_params = study.best_params
print("Best parameters:", best_params)

if not isinstance(best_params, dict):
    print("⚠️ Error: best_params is not a valid dictionary. Fetching another!")
    try:
        with open(BEST_PARAMS_PATH, "r") as f:
            best_params = json.load(f)
        print("✅ Parameters corrected:", best_params)
    except Exception as e:
        print(f"❌ Error reloading best_params: {e}")
        best_params = {}  # Avoid a crash

# Instead of training again, load the saved model (was before, but problems arose with saving histories)
# Rebuild the final GAN model with the best parameters

loss_final, history, gan, _ = train(
    mse_weight=best_params['mse_weight'],
    cosine_weight=best_params['cosine_weight'],
    experimental_mse_weight=best_params['experimental_mse_weight'],
    experimental_cosine_weight=best_params['experimental_cosine_weight'],
    derivative_weight=best_params['derivative_weight'], 
    huber_weight=best_params['huber_weight'],
    experimental_huber_weight=best_params['experimental_huber_weight'],
    bands_weight=best_params['bands_weight'],
    penalty_weight=best_params['penalty_weight'],
    angles_mean=angles_mean,       
    angles_std=angles_std,         
    angles_exp_mean=angles_exp_mean,    
    angles_exp_std=angles_exp_std,     
    glr=1e-4,
    dlr=1e-5,
    epochs=epochs,             
    train_dataset_sim=train_dataset_sim, 
    test_dataset_sim=test_dataset_sim,
    train_dataset_exp=train_dataset_exp, 
    test_dataset_exp=test_dataset_exp,
    gweight=best_params['generator_weight'],
    wavelengths=wavelengths_exp,
    gen_num_lstm_layers=best_params['gen_num_lstm_layers'],
    gen_lstm_units=best_params['gen_lstm_units'],
    gen_num_dense_layers=best_params['generator_num_dense_layers'],
    gen_dense_units=best_params['generator_dense_units'],
    disc_num_dense_layers=best_params['discriminator_num_dense_layers'],
    disc_dense_units=best_params['discriminator_dense_units'],
    gen_activation=best_params['gen_activation'],
    disc_activation=best_params['disc_activation']
)

#General load tests:
if gan.generator and gan.discriminator:
    print("Generator and Discriminator loaded and ready to use!")
elif gan.generator:
    print("Generator loaded, but discriminator not found.")
elif gan.discriminator:
    print("Discriminator loaded, but generator not found.")
else:
    print("Error loading saved models.")

# Save the best model and hyperparameters!
best_trial = study.best_trial
save_model_and_params(gan.generator, gan.discriminator, {
    "mse_weight": best_params.get('mse_weight', 2.0),  
    "cosine_weight": best_params.get('cosine_weight', 2.0),
    "experimental_mse_weight": best_params.get('experimental_mse_weight', 10.0),
    "experimental_cosine_weight": best_params.get('experimental_cosine_weight', 10.0),
    "derivative_weight": best_params.get('derivative_weight', 10.0),
    "huber_weight": best_params.get('huber_weight', 10.0),
    "experimental_huber_weight": best_params.get('experimental_huber_weight', 10.0),
    "bands_weight": best_params.get('bands_weight', 10.0),
    "penalty_weight": best_params.get('penalty_weight', 10.0), 
    "generator_lr": 1e-4,
    "discriminator_lr": 1e-5,
    "generator_weight": best_params.get('generator_weight', 10.0),
    "generator_num_dense_layers": best_params.get('generator_num_dense_layers', 2),
    "generator_dense_units": best_params.get('generator_dense_units', 512),
    "generator_activation": best_params.get('gen_activation', 'relu'),
    "discriminator_num_dense_layers": best_params.get('discriminator_num_dense_layers', 2),
    "discriminator_dense_units": best_params.get('discriminator_dense_units', 512),
    "discriminator_activation": best_params.get('disc_activation', 'relu'),
    "gen_num_lstm_layers": best_params.get('gen_num_lstm_layers', 1),
    "gen_lstm_units": best_params.get('gen_lstm_units', 64)

}, BEST_MODEL_PATH, BEST_PARAMS_PATH)

# Print the best parameters directly
print("Best parameters found:")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

gan_generator = None  # Initialize variable to avoid reference error

if os.path.exists(BEST_MODEL_PATH):
    print("Loading the best saved model")
    gan_generator, gan_discriminator, best_params = load_full_model(BEST_MODEL_PATH, "best_discriminator.h5", BEST_PARAMS_PATH)

    if isinstance(best_params, dict):
        print("Parameters loaded correctly:", best_params)
    else:
        print("⚠️ Error: best_params is not a valid dictionary. Fetching!")

        with open(BEST_PARAMS_PATH, "r") as f:
            best_params = json.load(f)  # Reload manually
        print("Parameters corrected:", best_params)

    if gan_generator:
        print("Model loaded and ready to be used in training!")
        print(f"Parameters loaded: {best_params}")
else:
    print(f"Error: Model '{BEST_MODEL_PATH}' not found. Proceeding without loading.")

history_path = "best_history.json"

if os.path.exists(history_path):
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
        print("✅ Training history loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading best_history.json: {e}")
        history = None
else:
    print("⚠️ No training history found. Creating a new one")
    history = None  # Set to None if there is no saved history yet!

print("Hyperparameters available for visualization:", study.best_trial.params.keys())
print("Hyperparameters in the study:", study.best_trial.params.keys())

visualize_optuna_results(study)

visualize_results(
    gan, 
    test_dataset_sim, 
    test_dataset_exp, 
    angles_mean, 
    angles_std, 
    angles_exp_mean, 
    angles_exp_std, 
    history, 
    num_samples=10
)