import os
import tensorflow as tf
from typing import Tuple
import joblib
import numpy as np

class ZeroDCE(tf.keras.Model):
    def __init__(self, name: str = "DCE-net", filters: int = 32, iteration: int = 8, IMG_H: int = 384, IMG_W: int = 512, IMG_C: int = 3, **kwargs):
        super(ZeroDCE, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.iteration = iteration
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.IMG_C = IMG_C

        self.conv1 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.concat_3_4 = tf.keras.layers.Concatenate(axis=-1)
        self.conv5 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.concat_2_5 = tf.keras.layers.Concatenate(axis=-1)
        self.conv6 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.concat_1_6 = tf.keras.layers.Concatenate(axis=-1)
        self.a_map_conv = tf.keras.layers.Conv2D(self.iteration * 3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='tanh')

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x_concat_3_4 = self.concat_3_4([x3, x4])
        x_5 = self.conv5(x_concat_3_4)
        x_concat_2_5 = self.concat_2_5([x2, x_5])
        x_6 = self.conv6(x_concat_2_5)
        x_concat_1_6 = self.concat_1_6([x1, x_6])
        a_maps = self.a_map_conv(x_concat_1_6)

        a_maps_splited = tf.split(a_maps, self.iteration, axis=-1)
        le_img = inputs
        for a_map in a_maps_splited:
            le_img = le_img + a_map * (tf.square(le_img) - le_img)
        return le_img, a_maps

    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        spatial_consistency_loss: tf.keras.losses.Loss,
        exposure_control_loss: tf.keras.losses.Loss,
        color_constancy_loss: tf.keras.losses.Loss,
        illumination_smoothness_loss: tf.keras.losses.Loss,
        loss_weights: dict = {
            'spatial_consistency_w': 1.0,
            'exposure_control_w': 20.0,
            'color_constancy_w': 10.0,
            'illumination_smoothness_w': 400.0
        },
        **kwargs
    ):
        print("\nstarted compiling.....\n")
        super(ZeroDCE, self).compile(**kwargs)
        self.optimizer = optimizer
        self.spatial_consistency_loss = spatial_consistency_loss
        self.exposure_control_loss = exposure_control_loss
        self.color_constancy_loss = color_constancy_loss
        self.illumination_smoothness_loss = illumination_smoothness_loss
        self.loss_weights = loss_weights

    def compute_losses(self, input_img: tf.Tensor, enhanced_img: tf.Tensor, a_maps: tf.Tensor) -> dict:
        '''
        Compute all zero reference DCE losses
        '''
        l_spa = self.loss_weights['spatial_consistency_w'] * self.spatial_consistency_loss(input_img, enhanced_img)
        
        # Assuming you need to compare enhanced_img against some target exposure
        exposure_target = tf.ones_like(enhanced_img)  # Example target; adjust as needed
        l_exp = self.loss_weights['exposure_control_w'] * self.exposure_control_loss(exposure_target, enhanced_img)
        
        l_col = self.loss_weights['color_constancy_w'] * self.color_constancy_loss(enhanced_img, enhanced_img)  # Adjusted call
        l_ill = self.loss_weights['illumination_smoothness_w'] * self.illumination_smoothness_loss(a_maps, a_maps)  # Adjusted call

        total_loss = l_spa + l_exp + l_col + l_ill

        return {
            'total_loss': total_loss,
            'spatial_consistency_loss': l_spa,
            'exposure_control_loss': l_exp,
            'color_constancy_loss': l_col,
            'illumination_smoothness_loss': l_ill
    }



    def get_compile_config(self):
        return {
            'optimizer': self.optimizer,
            'spatial_consistency_loss': self.spatial_consistency_loss,
            'exposure_control_loss': self.exposure_control_loss,
            'color_constancy_loss': self.color_constancy_loss,
            'illumination_smoothness_loss': self.illumination_smoothness_loss,
            'loss_weights': self.loss_weights
        }

    @classmethod
    def compile_from_config(cls, config):
        optimizer = config['optimizer']
        spatial_consistency_loss = config['spatial_consistency_loss']
        exposure_control_loss = config['exposure_control_loss']
        color_constancy_loss = config['color_constancy_loss']
        illumination_smoothness_loss = config['illumination_smoothness_loss']
        loss_weights = config['loss_weights']
        instance = cls()
        instance.compile(
            optimizer=optimizer,
            spatial_consistency_loss=spatial_consistency_loss,
            exposure_control_loss=exposure_control_loss,
            color_constancy_loss=color_constancy_loss,
            illumination_smoothness_loss=illumination_smoothness_loss,
            loss_weights=loss_weights
        )
        return instance

    @tf.function
    def train_step(self, inputs: tf.Tensor) -> dict:
        with tf.GradientTape() as tape:
            enhanced_img, a_maps = self(inputs)
            losses = self.compute_losses(inputs, enhanced_img, a_maps)

        gradients = tape.gradient(losses['total_loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return losses

    @tf.function
    def test_step(self, inputs: tf.Tensor) -> dict:
        enhanced_img, a_maps = self(inputs)
        val_losses = self.compute_losses(inputs, enhanced_img, a_maps)
        return val_losses

    def summary(self, plot: bool = False):
        x = tf.keras.Input(shape=(self.IMG_H, self.IMG_W, self.IMG_C))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='DCE-net')
        if plot:
            tf.keras.utils.plot_model(model, to_file='DCE-net.png', show_shapes=True, show_layer_names=True, rankdir='TB')
        return model.summary()

    def get_config(self):
        return {
            'filters': self.filters,
            'iteration': self.iteration
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == '__main__':
    import glob
    import cv2

    # Define image dimensions and other constants
    IMG_H, IMG_W, IMG_C = 384, 512, 3
    BATCH_SIZE = 16
    EPOCHS = 5

    # Paths
    X_folder = 'x2'  # Folder containing dark images
    Y_folder = 'y2'  # Folder containing bright images

    # Load and preprocess images
    def load_and_preprocess_image(path, target_h, target_w):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=IMG_C)
        image = tf.image.resize(image, [target_h, target_w])
        image = image / 255.0  # Normalize to [0, 1]
        return image

    def load_dataset(X_folder, Y_folder, batch_size):
        X_paths = sorted([os.path.join(X_folder, img) for img in os.listdir(X_folder)])
        Y_paths = sorted([os.path.join(Y_folder, img) for img in os.listdir(Y_folder)])

        X_images = [load_and_preprocess_image(path, IMG_H, IMG_W) for path in X_paths]
        Y_images = [load_and_preprocess_image(path, IMG_H, IMG_W) for path in Y_paths]

        # Split the dataset into training and validation sets
        X_train_images, X_val_images = X_images[:4300], X_images[4300:5000]
        Y_train_images, Y_val_images = Y_images[:4300], Y_images[4300:5000]

        X_train_dataset = tf.data.Dataset.from_tensor_slices(X_train_images)
        Y_train_dataset = tf.data.Dataset.from_tensor_slices(Y_train_images)
        X_val_dataset = tf.data.Dataset.from_tensor_slices(X_val_images)
        Y_val_dataset = tf.data.Dataset.from_tensor_slices(Y_val_images)

        train_dataset = tf.data.Dataset.zip((X_train_dataset, Y_train_dataset))
        val_dataset = tf.data.Dataset.zip((X_val_dataset, Y_val_dataset))

        train_dataset = train_dataset.shuffle(buffer_size=len(X_train_images)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    # Load the dataset
    print("Loading dataset...")
    train_dataset, val_dataset = load_dataset(X_folder, Y_folder, BATCH_SIZE)
    print("Dataset loaded successfully!")

    model = ZeroDCE(filters=32, iteration=8)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        spatial_consistency_loss=tf.keras.losses.MeanSquaredError(),
        exposure_control_loss=tf.keras.losses.MeanSquaredError(),
        color_constancy_loss=tf.keras.losses.MeanSquaredError(),
        illumination_smoothness_loss=tf.keras.losses.MeanSquaredError()
    )
    
print("Starting training...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Training loop
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        train_losses = model.train_step(x_batch_train)
        if step % 10 == 0:
            print(f"Training step {step}: Total loss = {train_losses['total_loss'].numpy():.4f}, "
                f"Spatial Consistency loss = {train_losses['spatial_consistency_loss'].numpy():.4f}, "
                f"Exposure Control loss = {train_losses['exposure_control_loss'].numpy():.4f}, "
                f"Color Constancy loss = {train_losses['color_constancy_loss'].numpy():.4f}, "
                f"Illumination Smoothness loss = {train_losses['illumination_smoothness_loss'].numpy():.4f}")

    # Validation loop
    for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        val_losses = model.test_step(x_batch_val)
        if step % 10 == 0:
            print(f"Validation step {step}: Total loss = {val_losses['total_loss'].numpy():.4f}, "
                f"Spatial Consistency loss = {val_losses['spatial_consistency_loss'].numpy():.4f}, "
                f"Exposure Control loss = {val_losses['exposure_control_loss'].numpy():.4f}, "
                f"Color Constancy loss = {val_losses['color_constancy_loss'].numpy():.4f}, "
                f"Illumination Smoothness loss = {val_losses['illumination_smoothness_loss'].numpy():.4f}")

print("Training completed! Saving the model...")

# Save the model as a joblib file
model_path = "Horus.keras"
model.save(model_path)
print(f"Model saved to {model_path}")