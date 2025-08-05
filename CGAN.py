import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import scipy.interpolate

# Load and preprocess data:

def load_and_preprocess_data(data_dir_sim, data_dir_exp_train, data_dir_exp_test, test_size=0.2):
    
    df_sim = pd.read_csv(data_dir_sim)

    df_sim.loc[df_sim.index > 103, df_sim.columns[1:]] = 0. #turn zeros after the band

    wavelengths_sim = df_sim.iloc[1:, 0].values.astype(float)  # Extract wavelengths from the first column, skipping header row.
    angles_sim = df_sim.columns.values[1:].astype(float)  # Extract angles from column header, excluding the wavelength label.
    transmittances_sim = df_sim.iloc[1:, 1:].values.astype(float).T # Transpose for correct access
    
    print("Angles shape sim:", angles_sim.shape)
    print("Transmittances shape sim:", transmittances_sim.shape)

    # Normalize simulated data
    angles_norm_sim = (angles_sim - angles_sim.mean()) / angles_sim.std()
    transmittances_norm_sim = transmittances_sim / np.max(transmittances_sim)

    # Split simulated data
    X_train_angle_sim, X_test_angle_sim, y_train_trans_sim, y_test_trans_sim = train_test_split(
        angles_norm_sim, transmittances_norm_sim, test_size=test_size, random_state=42
    )

    train_dataset_sim = tf.data.Dataset.from_tensor_slices(((X_train_angle_sim, tf.random.normal((len(X_train_angle_sim), 100))), y_train_trans_sim)).batch(20)
    test_dataset_sim = tf.data.Dataset.from_tensor_slices(((X_test_angle_sim, tf.random.normal((len(X_test_angle_sim), 100))), y_test_trans_sim)).batch(20)

    #---------------------------------------------------
    #---------------------------------------------------

#Split of experimental data:

    # Training data:
    df_exp_train = pd.read_csv(data_dir_exp_train)

    wavelengths_exp_train = df_exp_train.iloc[1:, 0].values.astype(float)  # Wavelengths
    angles_exp_train = df_exp_train.columns.values[1:].astype(float)  # Angles for training
    transmittances_exp_train = df_exp_train.iloc[1:, 1:].values.astype(float).T  # Transmittance matrix (transposed)

    print("Angles shape exp train:", angles_exp_train.shape)
    print("Transmittances shape exp train:", transmittances_exp_train.shape)

    # Normalization of experimental angles and transmittances (training)
    angles_norm_exp_train = (angles_exp_train - angles_exp_train.mean()) / angles_exp_train.std()
    transmittances_norm_exp_train = transmittances_exp_train / np.max(transmittances_exp_train)

    # Creating the dataset directly without splitting.
    train_dataset_exp = tf.data.Dataset.from_tensor_slices(
        ((angles_norm_exp_train, tf.random.normal((len(angles_norm_exp_train), 100))), transmittances_norm_exp_train)
    ).repeat(5).batch(20)

    # Test Data:
    df_exp_test = pd.read_csv(data_dir_exp_test)

    wavelengths_exp_test = df_exp_test.iloc[1:, 0].values.astype(float)  # Wavelengths
    angles_exp_test = df_exp_test.columns.values[1:].astype(float)  # Angles for testing/validation
    transmittances_exp_test = df_exp_test.iloc[1:, 1:].values.astype(float).T  # Transmittance matrix (transposed)

    print("Angles shape exp test:", angles_exp_test.shape)
    print("Transmittances shape exp test:", transmittances_exp_test.shape)

    # Normalization of experimental angles and transmittances (testing)
    angles_norm_exp_test = (angles_exp_test - angles_exp_train.mean()) / angles_exp_train.std()
    transmittances_norm_exp_test = transmittances_exp_test / np.max(transmittances_exp_test)

    # Creating the dataset directly for validation without split.
    test_dataset_exp = tf.data.Dataset.from_tensor_slices(
        ((angles_norm_exp_test, tf.random.normal((len(angles_norm_exp_test), 100))), transmittances_norm_exp_test)
    ).repeat(5).batch(20)

    print("Checking original and normalized angles:")
    print("Experimental original angles (train):", angles_exp_train)
    print("Experimental normalized angles (train):", angles_norm_exp_train)

    print("Experimental original angles (test):", angles_exp_test)
    print("Experimental normalized angles (test):", angles_norm_exp_test)

    return (
        train_dataset_sim, test_dataset_sim,
        train_dataset_exp, test_dataset_exp,
        angles_sim.mean(), angles_sim.std(),
        angles_exp_train.mean(), angles_exp_train.std(), wavelengths_exp_train
    )

    #---------------------------------------------------
    #---------------------------------------------------

# Function to create the generator model

def make_generator_model():
    angle_input = keras.layers.Input(shape=(1,))
    noise_input = keras.layers.Input(shape=(100,))
    
    generator_input_sequence = keras.layers.RepeatVector(100)(noise_input)
    lstm = keras.layers.LSTM(units=32, name="latent_lstm")(generator_input_sequence)

    merged = keras.layers.Concatenate()([angle_input, lstm])

    for i in range(5):  # 3 camadas densas
        merged = keras.layers.Dense(1792, activation='relu', name=f"gen_dense_{i}")(merged)
        merged = keras.layers.Dropout(0.1)(merged)
    
    output = keras.layers.Dense(125, activation='sigmoid')(merged)
    
    return keras.Model(inputs=[angle_input, noise_input], outputs=output)


# Function to create the discriminator model

def make_discriminator_model():
    angle_input = keras.layers.Input(shape=(1,))
    transmittance_input = keras.layers.Input(shape=(125,))
    merged = keras.layers.Concatenate()([angle_input, transmittance_input])
    x = merged
    for i in range(4):  
        x = keras.layers.Dense(384, activation='relu', name=f"disc_dense_{i}")(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)  # Final output

    return keras.Model(inputs=[angle_input, transmittance_input], outputs=x)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# Define the loss functions

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

    #Calculates MSE Loss with a 10% penalty for angles greater than 60 degrees (considering stepped angles).

    normalized_60 = (60 - angles_mean) / angles_std

    real_transmittances = tf.cast(real_transmittances, dtype=tf.float32)
    generated_transmittances = tf.cast(generated_transmittances, dtype=tf.float32)
    angles = tf.cast(angles, dtype=tf.float32)

    loss = tf.reduce_mean(tf.square(real_transmittances - generated_transmittances))

    penalty_mask = tf.where(angles > normalized_60, 1.3, 1.0)  #1.1 to increase loss by 10%
    loss = tf.reduce_mean(loss * penalty_mask)

    return loss

def cosine_similarity_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std):
    
    # Calculates cosine similarity loss with 10% penalty for angles greater than 60 degrees (considering stepped angles).

    normalized_60 = (60 - angles_mean) / angles_std

    real_transmittances = tf.cast(real_transmittances, dtype=tf.float32)
    generated_transmittances = tf.cast(generated_transmittances, dtype=tf.float32)
    angles = tf.cast(angles, dtype=tf.float32)
    real_normalized = tf.math.l2_normalize(real_transmittances, axis=1)
    generated_normalized = tf.math.l2_normalize(generated_transmittances, axis=1)

    loss = 1 - tf.reduce_mean(tf.reduce_sum(real_normalized * generated_normalized, axis=1))

    # Penalty for angles greater than 60 degrees staggered
    penalty_mask = tf.where(angles > normalized_60, 1.3, 1.0) 

    # Penalty test
    num_penalized = tf.reduce_sum(tf.cast(angles > normalized_60, tf.int32)).numpy()
    print(f"Number of penalized angles in this batch: {num_penalized}") 

    loss = tf.reduce_mean(loss * penalty_mask)

    return loss

def interpolate_to_reference(curves, source_wavelengths, target_wavelengths):
    """
    Interpolates each curve in 'curves' from the 'source_wavelengths' domain to 'target_wavelengths'.
    Returns a matrix with the same shape as (batch_size, len(target_wavelengths)).
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
    Find the starting and ending wavelengths where the transmittance is above 10% of the maximum value.
    """
    epsilon = 1e-6  # Small value to avoid division by zero
    max_vals = tf.reduce_max(y, axis=1, keepdims=True) + epsilon
    threshold = 0.1 * max_vals

    mask_above_threshold = tf.cast(y > threshold, tf.float32)

    # Indices of the first and last point above the threshold
    start_idx = tf.argmax(mask_above_threshold, axis=1, output_type=tf.int32)
    end_idx = tf.shape(y)[1] - 1 - tf.argmax(tf.reverse(mask_above_threshold, axis=[1]), axis=1, output_type=tf.int32)

    # Get the maximum valid index in the `x` vector
    max_index = tf.shape(x)[-1] - 1  # Last valid index

    # Fix possible out-of-bounds indexes
    start_idx = tf.clip_by_value(start_idx, 0, max_index)
    end_idx = tf.clip_by_value(end_idx, 0, max_index)

    # DEBUG: Inspect values before using `tf.gather()`
    if tf.executing_eagerly():
        print(f"DEBUG: x shape = {x.shape}")
        print(f"DEBUG: max_index = {max_index.numpy()}")
        print(f"DEBUG: start_idx = {start_idx.numpy()}, end_idx = {end_idx.numpy()}")

    x = tf.reshape(x, [-1])  # Transforms into a one-dimensional vector to avoid shape errors

    # Ensures that the indices are valid before using tf.gather()
    valid_start_idx = tf.where(start_idx <= max_index, start_idx, max_index)
    valid_end_idx = tf.where(end_idx <= max_index, end_idx, max_index)

    # Returns the corrected initial and final wavelengths
    return tf.gather(x, valid_start_idx), tf.gather(x, valid_end_idx)

def band_loss(real_transmittances, generated_transmittances, wavelengths):
    """
    Calculates the band loss between the real and generated transmittances based on the 10% maximum transmittance points.
    """
    # Converts to float32
    real_transmittances = tf.cast(real_transmittances, dtype=tf.float32)
    generated_transmittances = tf.cast(generated_transmittances, dtype=tf.float32)
    wavelengths = tf.cast(wavelengths, dtype=tf.float32)

    batch_size = tf.shape(real_transmittances)[0]
    wavelengths_real = tf.tile(tf.expand_dims(wavelengths, axis=0), [batch_size, 1])
    wavelengths_gen = tf.tile(tf.expand_dims(wavelengths, axis=0), [batch_size, 1]) # If it changes in the future, it will be different

    # If they are different, apply interpolation
    if not tf.reduce_all(tf.equal(wavelengths_real, wavelengths_gen)):
        generated_transmittances = interpolate_to_reference(generated_transmittances, wavelengths_gen, wavelengths_real)

    start_true, end_true = find_10_percent_max_wavelengths(wavelengths_real, real_transmittances)
    start_pred, end_pred = find_10_percent_max_wavelengths(wavelengths_real, generated_transmittances)

    # Print extracted values
    if tf.executing_eagerly():
        print(f"Start True Wavelengths: {start_true.numpy()}")
        print(f"Start Predicted Wavelengths: {start_pred.numpy()}")
        print(f"Diff Start: {np.abs(start_true.numpy() - start_pred.numpy())}")

        print(f"End True Wavelengths: {end_true.numpy()}")
        print(f"End Predicted Wavelengths: {end_pred.numpy()}")
        print(f"Diff End: {np.abs(end_true.numpy() - end_pred.numpy())}")

    # Normalization of the loss with the total wavelength range
    total_wavelength_range = tf.reduce_max(wavelengths) - tf.reduce_min(wavelengths) + 1e-6  # Avoid division by zero

    # Calculates the MSE between the start and end values
    start_loss = tf.reduce_mean(tf.square(start_true - start_pred) / total_wavelength_range)
    end_loss = tf.reduce_mean(tf.square(end_true - end_pred) / total_wavelength_range)

    return start_loss + end_loss

def derivative_loss(real_transmittances, generated_transmittances, wavelengths, penalty_weight):
    
    #Calculates the loss based on the difference of the derivatives of the transmittance curves.

    # Converts to numpy float32 before creating the tensor
    real_transmittances = tf.cast(real_transmittances, dtype=tf.float32)
    generated_transmittances = tf.cast(generated_transmittances, dtype=tf.float32)
    wavelengths = tf.cast(wavelengths, dtype=tf.float32)

    batch_size = tf.shape(real_transmittances)[0]
    wavelengths_real = tf.tile(tf.expand_dims(wavelengths, axis=0), [batch_size, 1])
    wavelengths_gen = tf.tile(tf.expand_dims(wavelengths, axis=0), [batch_size, 1]) 

    # Align domains by interpolation if they are different
    if not tf.reduce_all(tf.equal(wavelengths_real, wavelengths_gen)):
        generated_transmittances = interpolate_to_reference(generated_transmittances, wavelengths_gen, wavelengths_real)

    # Ensure tensors have at least 2 dimensions
    if tf.rank(real_transmittances) == 1:
        real_transmittances = tf.expand_dims(real_transmittances, axis=0)

    if len(generated_transmittances.shape) == 1:
        generated_transmittances = tf.expand_dims(generated_transmittances, axis=0)

    if len(wavelengths.shape) == 1:
        wavelengths = tf.expand_dims(wavelengths, axis=0)

    # Avoid slicing out of bounds
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

    # Adjust the penalty weight (tunable hyperparameter)
    penalty_weight = 32.06519066723878 # Test values like 3, 5, or 10 to adjust smoothness

    return first_derivative_loss + penalty_weight * second_derivative_penalty

def huber_loss(y_true, y_pred, angles, angles_mean, angles_std, delta=1.0):
    """
    Huber Loss: A combination of MSE and MAE that avoids extreme penalties for outliers.
    Heavily penalizes small errors and smooths out large errors.
    """
    # 🔹 Converts to the same type to avoid dtype error
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    angles = tf.cast(angles, dtype=tf.float32)

    error = y_true - y_pred
    is_small_error = tf.abs(error) < delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)

    loss = tf.where(is_small_error, squared_loss, linear_loss)

    # Penalty for angles greater than 60°
    normalized_60 = (60 - angles_mean) / angles_std
    penalty_mask = tf.where(angles > normalized_60, 1.2, 1.0)

    # 🔹 Expands `penalty_mask` to match the dimensions of `loss`
    return tf.reduce_mean(loss * tf.expand_dims(penalty_mask, axis=-1))

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# Define the ConditionalGAN model

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, mse_weight=13.234974116804501, cosine_weight=4.5603749046909368, experimental_mse_weight= 29.538860772664147, experimental_cosine_weight=4.559987640293828, derivative_weight=5.36731744496462,
    huber_weight=52.72008694123017, bands_weight=32.699706093991395, experimental_huber_weight=3):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.experimental_mse_weight = experimental_mse_weight
        self.experimental_cosine_weight = experimental_cosine_weight
        self.derivative_weight = derivative_weight  
        self.huber_weight = huber_weight 
        self.bands_weight = bands_weight
        self.experimental_huber_weight = experimental_huber_weight

    def compile(self, d_optimizer, g_optimizer):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
    
    def train_step(self, angles, noise, real_transmittances, experimental_angles, experimental_transmittances, noise_exp, wavelengths):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # Standard GAN Loss Calculation
            generated_transmittances = self.generator([angles, noise], training=True)
            experimental_gen_transmittances = self.generator([experimental_angles, noise_exp], training=True)

            # Debug to check indices before calculating band_loss
            start_true, end_true = find_10_percent_max_wavelengths(wavelengths, real_transmittances)
            start_pred, end_pred = find_10_percent_max_wavelengths(wavelengths, generated_transmittances)

            if tf.executing_eagerly():
                print(f"Start True Wavelengths: {start_true.numpy()}")
                print(f"End True Wavelengths: {end_true.numpy()}")
                print(f"Start Predicted Wavelengths: {start_pred.numpy()}")
                print(f"End Predicted Wavelengths: {end_pred.numpy()}")

            # Calculates the new derivative loss
            deriv_loss_value = derivative_loss(
                experimental_transmittances,
                experimental_gen_transmittances,
                wavelengths,
                self.derivative_weight 
            )
            
            band_loss_value = band_loss(experimental_transmittances, experimental_gen_transmittances, wavelengths)
            experimental_transmittances = tf.cast(experimental_transmittances, dtype=tf.float32)
            huber_loss_value = huber_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)
            experimental_huber_loss = huber_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_exp_mean, angles_exp_std)

            # Calculation of standard CGAN losses
            real_output = self.discriminator([angles, real_transmittances], training=True)
            fake_output = self.discriminator([angles, generated_transmittances], training=True)
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            mse_loss_value = mse_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)
            cosine_loss_value = cosine_similarity_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)

            # Experimental Data Loss Calculation
            experimental_mse_loss = mse_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_mean, angles_std)
            experimental_cosine_loss = cosine_similarity_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_exp_mean, angles_exp_std)

            # Combine Losses with Weights
            total_gen_loss = 1.080908969686041*gen_loss + self.mse_weight * mse_loss_value + \
                    self.cosine_weight * cosine_loss_value + \
                    self.experimental_mse_weight * experimental_mse_loss + \
                    self.experimental_cosine_weight * experimental_cosine_loss + \
                    self.derivative_weight * deriv_loss_value + \
                    self.huber_weight * huber_loss_value + \
                    self.bands_weight * band_loss_value + self.experimental_huber_weight * experimental_huber_loss

        gradients_of_generator = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Logs and returns
        print(f"Derivative Loss: {deriv_loss_value.numpy():.4f}")
        print(f"Huber Loss (Train): {huber_loss_value.numpy():.4f}") 
        print(f"Huber Loss (Validation): {experimental_huber_loss.numpy():.4f}") 
        print(f"Generated Transmittances Shape: {generated_transmittances.shape}")
        print(f"Test Step: Experimental Transmittances Shape: {experimental_transmittances.shape}")
        print(f"band_loss_value: {band_loss_value.numpy():.4f}")

        return {
            'd_loss': disc_loss,
            'g_loss': gen_loss,
            'mse_loss': mse_loss_value,
            'cosine_loss': cosine_loss_value,
            'experimental_mse_loss': experimental_mse_loss,
            'experimental_cosine_loss': experimental_cosine_loss,
            'derivative_loss': deriv_loss_value,
            'train_huber_loss': huber_loss_value,  
            'experimental_huber_loss': experimental_huber_loss ,
            'band_loss': band_loss_value            

        }

    def test_step(self, angles, noise, real_transmittances, experimental_angles, experimental_transmittances, noise_exp, wavelengths):
        # Standard GAN Loss Calculation

        generated_transmittances = self.generator([angles, noise], training=False)
        real_output = self.discriminator([angles, real_transmittances], training=False)
        fake_output = self.discriminator([angles, generated_transmittances], training=False)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        mse_loss_value = mse_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)
        cosine_loss_value = cosine_similarity_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)

        # Initialize all metrics to avoid inconsistency
        experimental_mse_loss = 0.0
        experimental_cosine_loss = 0.0
        derivative_loss_value = 0.0
        band_loss_value = 0.0
        huber_loss_value = 0.0
        experimental_huber_loss = 0.0

        if experimental_transmittances is not None:
            
            # Experimental Data Loss Calculation
            experimental_transmittances = tf.cast(experimental_transmittances, dtype=tf.float32)
            experimental_gen_transmittances = self.generator([experimental_angles, noise_exp], training=False)
            
            # Calculate the loss for experimental data only
            experimental_mse_loss = mse_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_mean, angles_std)
            experimental_cosine_loss = cosine_similarity_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_exp_mean, angles_exp_std)

            # Calculates the new derivative loss
            derivative_loss_value = derivative_loss(
                experimental_transmittances,
                experimental_gen_transmittances,
                wavelengths,
                self.derivative_weight  
            )

            band_loss_value = band_loss(experimental_transmittances, experimental_gen_transmittances, wavelengths)
            huber_loss_value = huber_loss(real_transmittances, generated_transmittances, angles, angles_mean, angles_std)
            experimental_huber_loss = huber_loss(experimental_transmittances, experimental_gen_transmittances, experimental_angles, angles_exp_mean, angles_exp_std)


        return {
            'd_loss': disc_loss,
            'g_loss': gen_loss,
            'mse_loss': mse_loss_value,
            'cosine_loss': cosine_loss_value,
            'experimental_mse_loss': experimental_mse_loss,
            'experimental_cosine_loss': experimental_cosine_loss,
            'derivative_loss': derivative_loss_value,
            'train_huber_loss': huber_loss_value,
            'experimental_huber_loss': experimental_huber_loss,
            'band_loss': band_loss_value

        }

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# Training Loop

data_dir_sim = 'C:/Users/Usuario/Desktop/IA em dados Óticos de materiais/trasmittance_newRecipe_500samples.csv' 
data_dir_exp_train = 'C:/Users/Usuario/Desktop/IA em dados Óticos de materiais/data_exp_train_four_cluster0.csv'
data_dir_exp_test = 'C:/Users/Usuario/Desktop/IA em dados Óticos de materiais/data_exp_test_four_cluster0.csv' 
train_dataset_sim, test_dataset_sim, train_dataset_exp, test_dataset_exp, angles_mean, angles_std, angles_exp_mean, angles_exp_std, wavelengths_exp = load_and_preprocess_data(data_dir_sim, data_dir_exp_train, data_dir_exp_test)

generator = make_generator_model()
discriminator = make_discriminator_model()
gan = ConditionalGAN(discriminator, generator)

# Definition of exponential decay
lr_schedule_g = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,  # Initial learning rate
    decay_steps=75,              # Every 75 steps, the LR is adjusted
    decay_rate=0.95,             # LR multiplier per epoch
    staircase=True               # Staircase decay (discrete adjustment)
)

lr_schedule_d = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5, 
    decay_steps=75, 
    decay_rate=0.95,
    staircase=True
)

# Creating optimizers with ExponentialDecay
d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_d)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_g)

gan.compile(d_optimizer, g_optimizer)

epochs = 90
history = {'d_loss': [], 'g_loss': [], 'mse_loss': [], 'cosine_loss': [], 'experimental_mse_loss' : [], 'experimental_cosine_loss': [],
           'val_d_loss': [], 'val_g_loss': [], 'val_mse_loss': [], 'val_cosine_loss': [],\
              'val_experimental_mse_loss' : [], 'val_experimental_cosine_loss': [], \
                    'derivative_loss': [], 'val_derivative_loss': [], \
                        'train_huber_loss': [], 'experimental_huber_loss': [],\
                            'band_loss':[],'val_band_loss': []}


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
            wavelengths_exp 
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
            wavelengths_exp
        )

    # History Update
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
    history['derivative_loss'].append(train_loss['derivative_loss'].numpy())
    history['val_derivative_loss'].append(test_loss['derivative_loss'].numpy())
    history['train_huber_loss'].append(train_loss['train_huber_loss'].numpy())
    history['experimental_huber_loss'].append(test_loss['experimental_huber_loss'].numpy())
    history['band_loss'].append(train_loss['band_loss'].numpy())
    history['val_band_loss'].append(test_loss['band_loss'].numpy())

#  Saving the updated CSV at the end of each epoch

    # Create a DataFrame with the updated history and add the 'epoch' column
    df_history = pd.DataFrame(history)
    df_history['epoch'] = list(range(1, len(df_history) + 1))  # Adds the epoch numbering correctly

    # Define the path to save the CSV
    csv_save_path = os.path.join(os.getcwd(), "training_results_4.csv")

    # Save to CSV at each epoch to prevent data loss
    df_history.to_csv(csv_save_path, index=False)

    print(f" History saved in CSV after the season {epoch+1}: {csv_save_path}")

    print(f'Epoch {epoch+1}, d_loss: {train_loss["d_loss"].numpy():.4f}, \
          g_loss: {train_loss["g_loss"].numpy():.4f}, \
            mse_loss: {train_loss["mse_loss"].numpy():.4f}, \
                cosine_loss: {train_loss["cosine_loss"].numpy():.4f}, \
                    exp_mse: {train_loss["experimental_mse_loss"].numpy():.4f}, \
                        exp_cos: {train_loss["experimental_cosine_loss"].numpy():.4f}')
    
    print(f'Epoch {epoch+1}, val_d_loss: {test_loss["d_loss"].numpy():.4f}, val_g_loss: {test_loss["g_loss"].numpy():.4f}, val_mse_loss: {test_loss["mse_loss"].numpy():.4f}, val_cosine_loss: {test_loss["cosine_loss"].numpy():.4f}, val_exp_mse: {test_loss["experimental_mse_loss"].numpy():.4f}, val_exp_cos: {test_loss["experimental_cosine_loss"].numpy():.4f}')

# Create directory to save models if it doesn't exist
save_dir = os.getcwd()  # Works in any environment
generator_save_path = os.path.join(save_dir, "generator_weightsCGAN.h5")
discriminator_save_path = os.path.join(save_dir, "discriminator_weightsCGAN.h5")

# Save the weights of the generator and discriminator
gan.generator.save_weights(generator_save_path)
gan.discriminator.save_weights(discriminator_save_path)

print(f"Generator weights saved at: {generator_save_path}")
print(f"Discriminator weights saved at: {discriminator_save_path}")

# CSV with graph plotting information

# Load the real angles from the validation CSV
csv_real = "C:/Users/Usuario/Desktop/IA em dados Óticos de materiais/data_exp_test_four_cluster0.csv"
df_real = pd.read_csv(csv_real)

# Extract the angles from the real CSV (ignoring the first column which is wavelength)
real_angles = df_real.columns[1:].astype(float).to_numpy()
print(f" Real angles extracted from CSV: {real_angles}")

#  Normalize the angles using the same scale as training
angles_normalized = (real_angles - angles_exp_mean) / angles_exp_std
angles_normalized = angles_normalized.reshape(-1, 1) 

# Create random noise for each real angle
num_samples = len(real_angles)
noise = np.random.normal(0, 1, size=(num_samples, 100))

# Generate the curves with the already trained model
generated_transmittances = gan.generator.predict([angles_normalized, noise])
print(f"Shape das curvas geradas: {generated_transmittances.shape}")

# Create an array with the wavelengths used in training
wavelengths = wavelengths_exp 

global_min = np.min(generated_transmittances)
global_max = np.max(generated_transmittances)

def normalize_curve(curve):
    return (curve - global_min) / (global_max - global_min) if (global_max - global_min) != 0 else curve

# Apply global normalization to the generated curves
generated_transmittances = np.array([normalize_curve(curve) for curve in generated_transmittances])

# Testing CSV

# Check the size of the wavelength array and the generated curves
print(f"Tamanho de wavelengths: {len(wavelengths)}")
print(f"Tamanho de generated_transmittances: {generated_transmittances.shape}")

# Load the real wavelengths to ensure alignment
wavelengths_real = df_real.iloc[:, 0].values 

# Ensure that the generated wavelengths are identical to the real ones
if len(wavelengths_real) != generated_transmittances.shape[1]:
    print(f" Mismatch! Adjusting the size of the generated curves to {len(wavelengths_real)} points...")
    min_length = min(len(wavelengths_real), generated_transmittances.shape[1])
    wavelengths_real = wavelengths_real[:min_length] 
    generated_transmittances = generated_transmittances[:, :min_length] 

# Create a DataFrame where:
df_curves = pd.DataFrame({"Wavelength": wavelengths_real})  # Use the real wavelengths

# Add the generated transmittances in the columns corresponding to the real angles
for i, angle in enumerate(real_angles):
    df_curves[f"Angle_{angle}"] = generated_transmittances[i]

# Save the updated CSV correctly
csv_save_path = "transmittance_curves_4_2.csv"
df_curves.to_csv(csv_save_path, index=False)

print(f" Updated CSV saved correctly with the same wavelengths as the actual data: {csv_save_path}")

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#  Visualization 

def denormalize(data, mean, std):
    return data * std + mean

def visualize_results(gan, test_dataset, exp_dataset, angles_mean, angles_std, angles_exp_mean, angles_exp_std, num_samples=20):
    (angles, noise), real_transmittances = next(iter(test_dataset.take(1)))
    (angles_exp, noise_exp), exp_transmittances = next(iter(exp_dataset.take(1)))

    # Generate curves with the model
    generated_transmittances = gan.generator.predict([angles, noise])
    generated_transmittances_exp = gan.generator.predict([angles_exp, noise_exp])

    # Order experimental angles to avoid misalignment
    sorted_indices = np.argsort(angles_exp.numpy().flatten())
    angles_exp_sorted = angles_exp.numpy().flatten()[sorted_indices]
    exp_transmittances_sorted = exp_transmittances.numpy()[sorted_indices]
    generated_transmittances_exp_sorted = generated_transmittances_exp[sorted_indices]

    # Fix number of samples to avoid index overflow
    num_samples = min(num_samples, len(angles_exp_sorted))

    # Define grid dimensions for the plots
    ncols = 4
    nrows = (num_samples + ncols - 1) // ncols

    # Create the first visualization (Model vs. Generated Data)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))

    for i in range(num_samples):
        row = i // ncols
        col = i % ncols

        denorm_real_angle = denormalize(angles[i].numpy(), angles_mean, angles_std)
        denorm_real_transmittance = real_transmittances[i]
        denorm_gen_transmittance = generated_transmittances[i]

        axes[row, col].plot(denorm_real_transmittance, label=f'Real (Angle: {denorm_real_angle:.2f})')
        axes[row, col].plot(denorm_gen_transmittance, label='Generated')
        axes[row, col].set_xlabel('Wavelength Point')
        axes[row, col].set_ylabel('Transmittance')
        axes[row, col].set_title(f'Sample {i+1}')
        axes[row, col].legend()

    # Remover gráficos vazios
    for j in range(num_samples, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    fig.suptitle('Model vs. Generated Transmittance Curves', fontsize=16)
    plt.tight_layout()
    plt.savefig('./ModelvsGeneratedTransmittance_Curves_Sorted.png')

    # Create the second view (Real vs. Generated with ordered angles)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))

    for i in range(num_samples):
        row = i // ncols
        col = i % ncols

        denorm_exp_angle = denormalize(angles_exp_sorted[i], angles_exp_mean, angles_exp_std)
        denorm_exp_transmittance = exp_transmittances_sorted[i]
        denorm_gen_transmittance = generated_transmittances_exp_sorted[i]

        axes[row, col].plot(denorm_exp_transmittance, label=f'Real (Angle: {denorm_exp_angle:.2f})')
        axes[row, col].plot(denorm_gen_transmittance, label='Generated')
        axes[row, col].set_xlabel('Wavelength Point')
        axes[row, col].set_ylabel('Transmittance')
        axes[row, col].set_title(f'Sample {i+1}')
        axes[row, col].legend()

    # Remove empty plots
    for j in range(num_samples, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    fig.suptitle('Real vs. Generated Transmittance Curves (Sorted)', fontsize=16)
    plt.tight_layout()
    plt.savefig('./RealvsGeneratedTransmittance_Sorted.png')

visualize_results(gan, test_dataset_sim, test_dataset_exp, angles_mean, angles_std, angles_exp_mean, angles_exp_std)

# Band loss
plt.figure(figsize=(12, 8))
plt.plot(history['band_loss'], label='Band Loss (Training)')
plt.plot(history['val_band_loss'], "--", label='Band Loss (Validation)')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Band Loss Over Training')
plt.legend()
plt.savefig('./BandLoss.png')  
plt.show()

# First Derivative Loss
plt.figure(figsize=(12, 8))
plt.plot(history['derivative_loss'], label='Derivative Loss (Training)')
plt.plot(history['val_derivative_loss'], "--", label='Derivative Loss (Validation)')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Derivative Loss Over Training')
plt.legend()
plt.savefig('./DerivativeLoss23.png')  
plt.show()

# Huber Loss Chart
plt.figure(figsize=(12, 8))
plt.plot(history['train_huber_loss'], label='Huber Loss (Training)')
plt.plot(history['experimental_huber_loss'], "--", label='Huber Loss (Validation)')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Huber Loss Over Training')
plt.legend()
plt.savefig('./HuberLoss.png') 
plt.show()

#-------------------------------------------------------------------

plt.figure(figsize=(12, 8))
plt.plot(history['d_loss'], label='Discriminator Loss')
plt.plot(history['g_loss'], label='Generator Loss')
plt.plot(history['mse_loss'], label='MSE Loss')
plt.plot(history['cosine_loss'], label='Cosine Loss')
plt.plot(history['experimental_mse_loss'], label='EXPERIMENTAL_MSE Loss')
plt.plot(history['experimental_cosine_loss'], label='EXPERIMENTAL_Cosine Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.savefig('./Training Losses23.png')

plt.figure(figsize=(12, 8))
plt.plot(history['val_d_loss'], label='Val Discriminator Loss')
plt.plot(history['val_g_loss'], label='Val Generator Loss')
plt.plot(history['val_mse_loss'], label='Val MSE Loss')
plt.plot(history['val_cosine_loss'], label='Val Cosine Loss')
plt.plot(history['val_experimental_mse_loss'], label='EXPERIMENTAL_MSE Loss')
plt.plot(history['val_experimental_cosine_loss'], label='EXPERIMENTAL_Cosine Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Losses')
plt.legend()
plt.savefig('./ValidationLosses23.png')


plt.figure(figsize=(12, 8))
plt.plot(history['val_mse_loss'], "--", label='Val MSE Loss')
plt.plot(history['val_cosine_loss'], "--", label='Val Cosine Loss')
plt.plot(history['val_experimental_mse_loss'], "--", label='val_EXPERIMENTAL_MSE Loss')
plt.plot(history['val_experimental_cosine_loss'], "--", label='val_EXPERIMENTAL_Cosine Loss')

plt.plot(history['mse_loss'], label='MSE Loss')
plt.plot(history['cosine_loss'], label='Cosine Loss')
plt.plot(history['experimental_mse_loss'], label='EXPERIMENTAL_MSE Loss')
plt.plot(history['experimental_cosine_loss'], label='EXPERIMENTAL_Cosine Loss')

plt.plot(history['experimental_mse_loss'] + history['experimental_cosine_loss'], label='SUM_EXPERIMENTAL_Loss')
plt.plot(history['val_experimental_mse_loss'] + history['val_experimental_cosine_loss'], label='SUM_Exp_validation_Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Losses')
plt.legend()
plt.savefig('./ValidationLosses_ALL23.png')

#Total Validation Losses
# This plot combines all experimental losses for a comprehensive view

plt.figure(figsize=(12, 8))

plt.plot(history['experimental_mse_loss'] + history['experimental_cosine_loss'], label='SUM_EXPERIMENTAL_Loss')
plt.plot(history['val_experimental_mse_loss'] + history['val_experimental_cosine_loss'], label='SUM_Exp_validation_Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Losses_total')
plt.legend()
plt.savefig('./ValidationLosses_total23_500épocas.png')