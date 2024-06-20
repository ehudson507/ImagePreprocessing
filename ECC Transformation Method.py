#This code is simply checking the transformation matrix. It takes the black and white mask TIF files as input and outputs the transformation.
#Also outputs visual confirmation


import cv2
import numpy as np
import glob
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(os.path.join(folder, '*.tif')):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            images.append(binary_img)
    return images

def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist

def histogram_intersection(hist1, hist2):
    return np.sum(np.minimum(hist1, hist2))

def find_optimal_base_image(images):
    histograms = [calculate_histogram(img) for img in images]
    num_images = len(images)
    scores = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(num_images):
            if i != j:
                scores[i][j] = histogram_intersection(histograms[i], histograms[j])
    
    avg_scores = scores.mean(axis=1)
    optimal_index = np.argmax(avg_scores)
    return images[optimal_index]

def calculate_centroid(image):
    moments = cv2.moments(image)
    if moments['m00'] != 0:
        centroid_x = int(moments['m10'] / moments['m00'])
        centroid_y = int(moments['m01'] / moments['m00'])
        return centroid_x, centroid_y
    else:
        return None, None

def pre_align_images(images, base_centroid):
    pre_aligned_images = []
    for img in images:
        centroid_x, centroid_y = calculate_centroid(img)
        if centroid_x is not None and centroid_y is not None:
            translation_x = base_centroid[0] - centroid_x
            translation_y = base_centroid[1] - centroid_y
            rows, cols = img.shape
            M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
            pre_aligned_img = cv2.warpAffine(img, M, (cols, rows))
            pre_aligned_images.append(pre_aligned_img)
        else:
            print("Unable to calculate centroid for an image.")
    return pre_aligned_images

def register_images(base_image, images):
    registered_images = []
    transformation_matrices = []
    base_image = base_image.astype(np.float32)
    for img in images:
        img_float = img.astype(np.float32)
        
        # Define the motion model
        warp_mode = cv2.MOTION_AFFINE
        
        # Initialize the matrix to identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Number of iterations (reduced for pre-aligned images)
        number_of_iterations = 250
        
        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, 1e-10)
        
        # Run the ECC algorithm to find the warp matrix
        cc, warp_matrix = cv2.findTransformECC(base_image, img_float, warp_matrix, warp_mode, criteria)
        
        # Warp the current image to align with the base image
        height, width = base_image.shape
        registered_img = cv2.warpAffine(img, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        registered_images.append(registered_img)
        transformation_matrices.append(warp_matrix)
    
    return registered_images, transformation_matrices

def display_images(images, transformation_matrices, base_centroid):
    plt.figure(figsize=(8, 8))
    overlay_image = np.zeros_like(images[0], dtype=np.float32)
    for i, img in enumerate(images):
        # Extract transformation parameters from ECC
        ecc_matrix = transformation_matrices[i]
        ecc_translation_x = ecc_matrix[0, 2]
        ecc_translation_y = ecc_matrix[1, 2]
        ecc_scaling_x = np.sqrt(ecc_matrix[0, 0] ** 2 + ecc_matrix[1, 0] ** 2)
        ecc_scaling_y = np.sqrt(ecc_matrix[0, 1] ** 2 + ecc_matrix[1, 1] ** 2)
        ecc_rotation = np.arctan2(ecc_matrix[1, 0] / ecc_scaling_x, ecc_matrix[0, 0] / ecc_scaling_x) * 180 / np.pi

        # Extract translation from pre-alignment by centroid
        centroid_x, centroid_y = calculate_centroid(img)
        pre_alignment_translation_x = base_centroid[0] - centroid_x
        pre_alignment_translation_y = base_centroid[1] - centroid_y

        # Combine translations
        total_translation_x = pre_alignment_translation_x + ecc_translation_x
        total_translation_y = pre_alignment_translation_y + ecc_translation_y

        # Print combined transformation vector

        print(f'Total Translation: ({total_translation_x}, {total_translation_y})')
        print(f'ECC Translation: ({ecc_translation_x}, {ecc_translation_y})')
        print(f'Pre-alignment Translation: ({pre_alignment_translation_x}, {pre_alignment_translation_y})')
        print(f'ECC Scaling: ({ecc_scaling_x:.2f}, {ecc_scaling_y:.2f})')
        print(f'ECC Rotation: {ecc_rotation:.2f} degrees\n')
        
        # Display image with transparency
        plt.imshow(img, cmap='gray', alpha=0.5)
    plt.title('Overlay of Registered Images')
    plt.axis('off')
    plt.show()

def main():
    # Use tkinter to select folder
    root = tk.Tk()
    root.withdraw()  # Close the root window
    folder = filedialog.askdirectory(title="Select the folder containing TIFF images")
    
    if not folder:
        print("No folder selected.")
        return
    
    images = load_images_from_folder(folder)
    
    if not images:
        print("No images found in the folder.")
        return
    
    base_image = find_optimal_base_image(images)
    
    # Calculate centroid of the base image
    base_centroid = calculate_centroid(base_image)
    
    # Pre-align images by their centroid
    pre_aligned_images = pre_align_images(images, base_centroid)
    
    registered_images, transformation_matrices = register_images(base_image, pre_aligned_images)
    
    # Save registered images
    for i, img in enumerate(registered_images):
        cv2.imwrite(os.path.join(folder, f'registered_image_{i}.tiff'), img)
    
    # Display registered images using Matplotlib
    display_images(registered_images, transformation_matrices, base_centroid)

if __name__ == "__main__":
    main()
