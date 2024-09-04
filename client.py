import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple

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
        l_spa = self.loss_weights['spatial_consistency_w'] * self.spatial_consistency_loss(input_img, enhanced_img)
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


# Load the trained DCE-Net model
model_path = 'Horus1.keras'  # Update with your actual model filename
loaded_model = tf.keras.models.load_model(model_path, custom_objects={'ZeroDCE': ZeroDCE})

# Directories for input and output images
input_dir = 'enhanced_images'  # Folder with dark images
output_dir = 'new_enhanced_images'  # Folder where enhanced images will be saved
output_dir1 = 'reference_images'
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir1, exist_ok=True)

# Function to enhance the image
def enhance_image(image, model):
    # Resize the image to 384x512
    image = cv2.resize(image, (512, 384))  # OpenCV uses (width, height)
    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Predict enhanced image
    model_output = model.predict(image)
    
    # Check if model output is a tuple
    if isinstance(model_output, tuple):
        # Assuming the first element is the image enhancement result
        enhanced_image = model_output[0]
    else:
        enhanced_image = model_output
    
    # Debugging: Print the type and shape of the enhanced image
    print(f"Type of enhanced image: {type(enhanced_image)}")
    if isinstance(enhanced_image, np.ndarray):
        print(f"Shape of enhanced image: {enhanced_image.shape}")
    else:
        raise TypeError("The model output is not a numpy array or TensorFlow tensor.")
    
    # Ensure the output shape is correct
    if enhanced_image.ndim != 4 or enhanced_image.shape[0] != 1:
        raise ValueError("The model output shape is not as expected. Expected shape: [1, height, width, channels].")

    # Remove batch dimension and convert to TensorFlow tensor
    enhanced_image = np.squeeze(enhanced_image, axis=0)
    print(f"Shape of enhanced image after squeezing: {enhanced_image.shape}")
    
    if enhanced_image.ndim != 3:
        raise ValueError("The model output shape after squeezing is not as expected. Expected shape: [height, width, channels].")
    
    # Convert to TensorFlow tensor and resize
    enhanced_image = tf.convert_to_tensor(enhanced_image)
    enhanced_image = tf.image.resize(enhanced_image, (384, 512))
    
    # Convert back to numpy array and rescale to [0, 255]
    enhanced_image = enhanced_image.numpy()
    enhanced_image = (enhanced_image * 255).astype('uint8')
    return enhanced_image

# Load the trained DCE-Net model
model_path = 'Horus.keras'  # Update with your actual model filename
loaded_model = tf.keras.models.load_model(model_path, custom_objects={'ZeroDCE': ZeroDCE})

# Directories for input and output images
input_dir = 'enhanced_images'  # Folder with dark images
output_dir = 'new_enhanced_images'  # Folder where enhanced images will be saved
output_dir1 = 'reference_images'
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
j=0
# Process each image in the input directory
for img_name in os.listdir(input_dir):
    if j>5:
        break
    img_path = os.path.join(input_dir, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Can't load image {img_name}. Skipping...")
        continue
    
    enhanced_image = enhance_image(image, loaded_model)
    
    # Save the enhanced image
    output_path = os.path.join(output_dir, img_name)
    output_path1 = os.path.join(output_dir1, img_name)
    cv2.imwrite(output_path, enhanced_image)
    cv2.imwrite(output_path1, image)
    print(f"Enhanced image saved to {output_path}")
    j+=1

print("All images have been processed and enhanced!")