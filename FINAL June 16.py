#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 11:54:54 2024

@author: emma
"""

import os
import shutil
import re
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import openslide
import json
import glob
import platform
from tqdm import tqdm

# Function to get the directory from the user
def get_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory()
    return folder_selected

# Function to create mask and save it using the provided process_image method
def create_mask(imgName, path):
    try:
        # Read image using Openslide
        slide = openslide.OpenSlide(os.path.join(path, imgName))
        level = 5 
        width, height = slide.level_dimensions[level]
        image = slide.read_region((0, 0), level, (width, height))
        image = np.array(image.convert("L"))  # Convert to grayscale

        # Process image to generate mask
        image, mask = process_image(image)
        
        # Get the name of the main folder (parent folder)
        main_folder_name = os.path.basename(os.path.normpath(path))
        
        # Create new subfolder name for masks
        new_subfolder_name = os.path.join(path, main_folder_name + "_MASK")
        
        # Save mask to new subfolder
        os.makedirs(new_subfolder_name, exist_ok=True)
        mask_path = os.path.join(new_subfolder_name, imgName[:-5] + "_MASK.tif")
        cv2.imwrite(mask_path, mask)
    except Exception as e:
        print(f"Error processing {imgName}: {e}")

# Function to process image and generate mask
def process_image(image):
    # Adaptive Thresholding to give a clearer definition of an image
    image_adapt = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)

    # Blurring and otsu thresholding
    blur = cv2.GaussianBlur(image_adapt, (25, 25), 0)
    _, thresholded = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Generate contours and remove small contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours that touch the borders or have boundary >= 95% of image
    bound = get_boundary_coord(contours, image.shape[0], image.shape[1])

    filtered_contours = [contours[n] for n in range(len(contours)) if cv2.contourArea(contours[n]) >= 5000 and \
                         bound[n][0] >= 10 and bound[n][2] <= (image.shape[1] - 10) and \
                         bound[n][1] >= 10 and bound[n][3] <= (image.shape[0] - 10) and \
                         bound[n][4] <= 0.9 * image.shape[1] and bound[n][5] <= 0.9 * image.shape[0]]

    # Create image mask
    mask = np.ones(thresholded.shape, dtype=np.uint8) * 255
    mask = cv2.drawContours(mask, filtered_contours, -1, (0), thickness=-1)

    return image, mask

# Function to get boundary coordinates of contours
def get_boundary_coord(contours, image_row, image_column):
    bound = []    

    # Transform into x1, y1, x2, y2, w, h
    for cnt in contours:
        _tmp = cv2.boundingRect(cnt)
        bound.append((_tmp[0], _tmp[1], _tmp[0] + _tmp[2], _tmp[1] + _tmp[3], _tmp[2], _tmp[3], cv2.contourArea(cnt)))

    return bound


    return bound
# Function to get the base directory from the user
def get_base_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    base_dir = filedialog.askdirectory(title="Select Base Directory")
    return base_dir

def select_base_image(mask_path):
    masks = [file for file in os.listdir(mask_path) if file.endswith("_MASK.tif")]
    if masks:
        return os.path.join(mask_path, masks[0])  # Return the first mask as base image
    else:
        return None

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

def register_images(base_image, images, mouse_number, body_part):
    registered_images = []
    transformation_matrices = []
    base_image = base_image.astype(np.float32)
    
    for img in tqdm(images, desc=f"Pre-processing {mouse_number}_{body_part}"):
        try:
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
        
        except cv2.error as e:
            print(f"Image registration failed: {e}")
            print(f"Image file name: {img}")
            
            if transformation_matrices:
                # Use the closest transformation found so far
                registered_img = cv2.warpAffine(img, transformation_matrices[-1], (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                registered_images.append(registered_img)
                transformation_matrices.append(transformation_matrices[-1])
            else:
                # If no prior transformation matrix exists, use identity matrix
                print("No previous transformations found. Using identity matrix.")
                registered_img = cv2.warpAffine(img, np.eye(2, 3, dtype=np.float32), (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                registered_images.append(registered_img)
                transformation_matrices.append(np.eye(2, 3, dtype=np.float32))
    
    return registered_images, transformation_matrices

def generate_transformation_vector(base_image_path, mask_path):
    images = load_images_from_folder(mask_path)
    base_image = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)

    if len(images) == 1:
        # Only one mask image case
        transVector = {}
        transVector[os.listdir(mask_path)[0]] = {
            "Total Translation": [0.0, 0.0],
            "ECC Scaling": [1.0, 1.0],
            "ECC Rotation": "0.00 degrees"
        }
        return transVector
    
    base_centroid = calculate_centroid(base_image)
    pre_aligned_images = pre_align_images(images, base_centroid)
    
    # Extract mouse_number and body_part from mask_path
    mouse_number, body_part = os.path.basename(mask_path).split("_")[0], os.path.basename(mask_path).split("_")[1]
    
    # Register images with mouse_number and body_part
    registered_images, transformation_matrices = register_images(base_image, pre_aligned_images, mouse_number, body_part)
    
    transVector = {}
    for i, file in enumerate(os.listdir(mask_path)):
        if file.endswith("_MASK.tif"):
            if i < len(transformation_matrices):
                warp_matrix = transformation_matrices[i]
                # Calculate translation, scaling, and rotation
                translation_x = warp_matrix[0, 2]
                translation_y = warp_matrix[1, 2]
                scaling_x = np.sqrt(warp_matrix[0, 0] ** 2 + warp_matrix[1, 0] ** 2)
                scaling_y = np.sqrt(warp_matrix[0, 1] ** 2 + warp_matrix[1, 1] ** 2)
                # Calculate rotation
                rotation = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0]) * (180 / np.pi)
                
                # Convert to native Python types
                translation_x = float(translation_x)
                translation_y = float(translation_y)
                scaling_x = float(scaling_x)
                scaling_y = float(scaling_y)
                rotation = f"{float(rotation):.2f} degrees"
                
                # Store transformation parameters
                transVector[file] = {
                    "Total Translation": [translation_x, translation_y],
                    "ECC Scaling": [scaling_x, scaling_y],
                    "ECC Rotation": rotation
                }
            else:
                transVector[file] = None
    
    return transVector

# Function to create directory structure and move files to the destination directory
def organize_files(base_dir, processed_dir):
    # Regular expression pattern to match the file pattern
    pattern = re.compile(r'\[#[0-9]+\] Yi_#([0-9]+)_(\w+)_')

    # Iterate through all subdirectories and files in the base directory
    for root, dirs, files in os.walk(base_dir):
        print(f"Processing directory: {root}")
        for file in files:
            if file.endswith('.ndpi'):
                imgName = file
                match = pattern.search(imgName)
                if match:
                    mouse_number = match.group(1)
                    body_part = match.group(2)
                    # Create the target directory for processed images in the processed directory
                    processed_dir_path = os.path.join(processed_dir, f"{mouse_number}_{body_part}")
                    os.makedirs(processed_dir_path, exist_ok=True)
                    # Move the file to the target directory in the processed directory
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(processed_dir_path, file)
                    shutil.copy(source_file, target_file)
                    # Create mask for the image in processed directory
                    create_mask(file, processed_dir_path)

    # Generate transformation vectors and save as JSON files
    for root, dirs, files in os.walk(processed_dir):
        for dir in dirs:
            if dir.endswith("_MASK"):
                mask_path = os.path.join(root, dir)
                base_image = select_base_image(mask_path)
                if base_image:
                    # Extract the name of the main folder from the root directory
                    main_folder_name = os.path.basename(os.path.normpath(root))
                    
                    trans_vector = generate_transformation_vector(base_image, mask_path)
                    # Create transformed directory in the respective mouse number and body part folder
                    transformed_dir = os.path.join(root, main_folder_name + "_TRANSFORMED")
                    os.makedirs(transformed_dir, exist_ok=True)
                    # Output transformation vector with custom filename in the processed directory
                    json_filename = f"{main_folder_name}_transformation_vector.json"
                    with open(os.path.join(transformed_dir, json_filename), "w") as trans_output:
                        json.dump(trans_vector, trans_output, indent=4)
                else:
                    print(f"No masks found in {mask_path}")

    print("Part 1 Complete.Files have been organized, masks have been generated, and transformation vectors have been saved in the processed directory.")

def load_transformation_vectors(json_file):
    with open(json_file) as f:
        transformation_vectors = json.load(f)
    return transformation_vectors

def transform_image_mod3(image, transformation, level_factor):
    scale_x, scale_y = map(float, transformation["ECC Scaling"])
    translation_x, translation_y = map(float, transformation["Total Translation"])
    rotation = float(transformation["ECC Rotation"].split()[0])

    rows, cols = image.shape[:2]

    # Apply the level factor to the translations
    translation_x *= level_factor
    translation_y *= level_factor

    # Calculate the new dimensions of the image after transformation
    cos_theta = np.abs(np.cos(np.radians(rotation)))
    sin_theta = np.abs(np.sin(np.radians(rotation)))
    new_cols = int((rows * sin_theta) + (cols * cos_theta))
    new_rows = int((rows * cos_theta) + (cols * sin_theta))

    # Update the translation to center the image
    translation_x += (new_cols - cols) / 2
    translation_y += (new_rows - rows) / 2

    # Transformation matrix
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)  # Fix scale to 1
    M[0, 0] *= scale_x
    M[1, 1] *= scale_y
    M[0, 2] += translation_x
    M[1, 2] += translation_y

    # Apply transformation
    transformed_image = cv2.warpAffine(image, M, (new_cols, new_rows))

    return transformed_image

def match_file(file_name, transformation_vectors):
    # Extract numerical identifier from the file name
    numerical_identifier = re.search(r'\[#(\d+)\]', file_name)
    if numerical_identifier:
        numerical_identifier = numerical_identifier.group(1)
        
        # Search for a matching key in the transformation vectors
        for key in transformation_vectors.keys():
            if numerical_identifier in key:
                return key

    return None




def crop_image(image):
    # Convert image to grayscale
    if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
            gray_image = image  # Assuming input image is already grayscale


    # Adaptive Thresholding to give a clearer definition of an image
    image_adapt = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)

    # Blurring and Otsu thresholding
    blur = cv2.GaussianBlur(image_adapt, (25, 25), 0)
    _, thresholded = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Generate contours and remove small contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get image dimensions
    image_row, image_column = gray_image.shape

    # Get bounding rectangles
    bound = get_boundary_coord(contours, image_row, image_column)

    # Filter out contours that touch the borders or have a boundary >= 95% of image
    filtered_contours = [contours[n] for n in range(len(contours)) if cv2.contourArea(contours[n]) >= 5000 and \
                         bound[n][0] >= 10 and bound[n][2] <= (image_column - 10) and \
                         bound[n][1] >= 10 and bound[n][3] <= (image_row - 10) and \
                         bound[n][4] <= 0.9 * image_column and bound[n][5] <= 0.9 * image_row]

    if not filtered_contours:
        print("No valid contours found. Skipping cropping.")
        return image  # Return the original image if no valid contours are found

    # Find the most upper-left and bottom-right points
    min_x = min(contour[:,:,0].min() for contour in filtered_contours)
    min_y = min(contour[:,:,1].min() for contour in filtered_contours)
    max_x = max(contour[:,:,0].max() for contour in filtered_contours)
    max_y = max(contour[:,:,1].max() for contour in filtered_contours)

    # Crop the image to the bounding rectangle of the contour
    cropped_image = image[min_y:max_y, min_x:max_x]
    
    return cropped_image


def process_image_mod3(image_path, transformation, level_factor, output_folder, pbar=None):
    # Read NDPI image using OpenSlide
    ndpi_image = openslide.OpenSlide(image_path)
    if not ndpi_image:
        print(f"Unable to load NDPI image {image_path}. Skipping.")
        return

    # Read the whole slide image and convert to RGB
    ndpi_image_array = ndpi_image.read_region((0, 0), 0, ndpi_image.level_dimensions[0])
    ndpi_image_array = np.array(ndpi_image_array)[:, :, :3]  # Convert RGBA to RGB

    # Downsample the image for processing (factor of 5)
    downsample_factor = 5
    small_image = cv2.resize(ndpi_image_array, 
                             (ndpi_image_array.shape[1] // downsample_factor, 
                              ndpi_image_array.shape[0] // downsample_factor),
                             interpolation=cv2.INTER_AREA)

    # Transform the small image
    transformed_image = transform_image_mod3(small_image, transformation, level_factor)

    # Crop the transformed image using the improved cropping mechanism
    cropped_image = crop_image(transformed_image)

    # Upsample the image back to original size
    final_image = cv2.resize(cropped_image, 
                             (ndpi_image_array.shape[1], 
                              ndpi_image_array.shape[0]),
                             interpolation=cv2.INTER_CUBIC)

    # Extract filename from image path
    image_name = os.path.basename(image_path)

    # Determine the output folder dynamically based on the JSON file path
    json_folder = os.path.dirname(output_folder)
    output_path = os.path.join(json_folder, f"transformed_cropped_{image_name}.jpeg")
    cv2.imwrite(output_path, final_image)
    print(f"Cropped and transformed image saved as {output_path}")

    # Update progress bar
    if pbar:
        pbar.update(1)




def find_json_files(processed_dir):
    # Function to find all JSON files within nested subfolders
    json_files = []
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files




def main():
    # Get the base directory from the user
    base_dir = get_base_directory()

    if base_dir:
        # Create "Processed Images" directory on the desktop
        desktop_dir = ''
        if platform.system() == "Windows":
            desktop_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        elif platform.system() == "Darwin":  # macOS
            desktop_dir = os.path.join(os.path.expanduser('~'), 'Desktop')
        else:  # Assuming Linux
            desktop_dir = os.path.join(os.path.expanduser('~'), 'Desktop')

        processed_dir = os.path.join(desktop_dir, "Processed Images")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Organize files and process data in the "Processed Images" directory
        organize_files(base_dir, processed_dir)
        
        # Find all JSON files in the processed_dir
        json_files = find_json_files(processed_dir)
        
        if not json_files:
            print("No JSON files found in the Processed Images directory. Exiting.")
            return

        # Iterate through each JSON file
        for json_file in json_files:
            # Extract mouse_number and body_part from JSON filename
            json_filename = os.path.basename(json_file)
            mouse_number, body_part = json_filename.split("_")[0], json_filename.split("_")[1]
            level_factor = 1
            
            # Calculate total number of images (keys) in this JSON file
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                total_images = len(json_data)

            # Progress bar for each JSON vector
            print(f"Processing images for {json_filename}:")
            with tqdm(total=total_images, desc=f"Transforming images in {json_filename}", unit="image", leave=False) as pbar:
                # Iterate through NDPI files in the folder corresponding to this JSON file
                ndpi_folder_path = os.path.join(processed_dir, f"{mouse_number}_{body_part}")
                for root, dirs, files in os.walk(ndpi_folder_path):
                    for filename in files:
                        if filename.endswith(".ndpi"):
                            ndpi_file = os.path.join(root, filename)

                            # Load transformation vectors specific to this JSON file
                            transformation_vectors = load_transformation_vectors(json_file)

                            # Check if the transformation vector for the image exists in this JSON file
                            matching_key = match_file(filename, transformation_vectors)
                            if matching_key:
                                transformation = transformation_vectors[matching_key]

                                # Construct path to transformed folder (one subfolder deeper)
                                transformed_folder = os.path.join(root, f"{mouse_number}_{body_part}_TRANSFORMED")
                                if not os.path.exists(transformed_folder):
                                    os.makedirs(transformed_folder, exist_ok=True)

                                # Process the image and save in the transformed_folder
                                output_filename = f"transformed_cropped_{filename}.jpeg"
                                output_path = os.path.join(transformed_folder, output_filename)
                                process_image_mod3(ndpi_file, transformation, level_factor, output_path, pbar=pbar)
                            else:
                                print(f"No transformation vector found for {filename}. Skipping.")

        print("Processing complete.")

    else:
        print("No base directory selected.")

if __name__ == "__main__":
    main()


