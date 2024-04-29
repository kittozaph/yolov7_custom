import numpy as np
import cv2
import os

# Define the Gaussian noise parameters
mean = 0
stddev = 0.1

# Get the list of images in the dataset
image_list = os.listdir('images')

# Load the images and add noise to each image
for image_name in image_list:
    image = cv2.imread(f'images/{image_name}')
    noise = np.random.normal(mean, stddev, image.shape)
    noisy_image = image + noise
    cv2.imwrite(f'noisy_images/{image_name}', noisy_image)
