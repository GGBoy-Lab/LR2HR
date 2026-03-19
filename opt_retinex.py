import sys
import os
import cv2
import json
import retinex
from pathlib import Path
import numpy as np


def batch_process_images(input_folder, output_folder, config_file='config.json'):
    """
    Batch process images using all Retinex methods

    Args:
        input_folder: Input folder path
        output_folder: Output folder root path
        config_file: Configuration file path
    """
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist!")
        return

    # Read configuration file
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} does not exist!")
        return

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Get all image files from input folder
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    img_files = []

    for file in os.listdir(input_folder):
        if any(file.lower().endswith(ext) for ext in img_extensions):
            if file != '.gitkeep':  # Exclude .gitkeep file
                img_files.append(file)

    if len(img_files) == 0:
        print(f'No image files found in input folder {input_folder}.')
        return

    # Create output subfolders - include all methods
    methods = ['original', 'ssr', 'msr', 'msrcr', 'amsrcr', 'msrcp']
    output_paths = {}

    for method in methods:
        path = os.path.join(output_folder, method)
        os.makedirs(path, exist_ok=True)
        output_paths[method] = path

    # Process each image
    for img_name in img_files:
        print(f"Processing: {img_name}")

        # Read image
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Cannot read image: {img_path}")
            continue

        # Get filename without extension
        name_without_ext = Path(img_name).stem

        # Convert image to float type for SSR and MSR processing
        img_float = img.astype(np.float64) + 1.0

        try:
            # Single Scale Retinex (SSR)
            # Use the first sigma value from configuration
            sigma = config['sigma_list'][0] if config['sigma_list'] else 300
            img_ssr = retinex.singleScaleRetinex(img_float, sigma)
            # Normalize to 0-255 range
            for i in range(img_ssr.shape[2]):
                img_ssr[:, :, i] = (img_ssr[:, :, i] - np.min(img_ssr[:, :, i])) / \
                                   (np.max(img_ssr[:, :, i]) - np.min(img_ssr[:, :, i])) * 255
            img_ssr = np.uint8(np.minimum(np.maximum(img_ssr, 0), 255))

            # Multi-Scale Retinex (MSR)
            img_msr = retinex.multiScaleRetinex(img_float, config['sigma_list'])
            # Normalize to 0-255 range
            for i in range(img_msr.shape[2]):
                img_msr[:, :, i] = (img_msr[:, :, i] - np.min(img_msr[:, :, i])) / \
                                   (np.max(img_msr[:, :, i]) - np.min(img_msr[:, :, i])) * 255
            img_msr = np.uint8(np.minimum(np.maximum(img_msr, 0), 255))

            # MSRCR method
            img_msrcr = retinex.MSRCR(
                img,
                config['sigma_list'],
                config['G'],
                config['b'],
                config['alpha'],
                config['beta'],
                config['low_clip'],
                config['high_clip']
            )

            # Automated MSRCR method
            img_amsrcr = retinex.automatedMSRCR(
                img,
                config['sigma_list']
            )

            # MSRCP method
            img_msrcp = retinex.MSRCP(
                img,
                config['sigma_list'],
                config['low_clip'],
                config['high_clip']
            )

            # Save results to corresponding folders
            # Original image
            original_output_path = os.path.join(output_paths['original'], img_name)
            cv2.imwrite(original_output_path, img)

            # SSR result
            ssr_output_path = os.path.join(output_paths['ssr'], f"{name_without_ext}_ssr.jpg")
            cv2.imwrite(ssr_output_path, img_ssr)

            # MSR result
            msr_output_path = os.path.join(output_paths['msr'], f"{name_without_ext}_msr.jpg")
            cv2.imwrite(msr_output_path, img_msr)

            # MSRCR result
            msrcr_output_path = os.path.join(output_paths['msrcr'], f"{name_without_ext}_msrcr.jpg")
            cv2.imwrite(msrcr_output_path, img_msrcr)

            # Automated MSRCR result
            amsrcr_output_path = os.path.join(output_paths['amsrcr'], f"{name_without_ext}_amsrcr.jpg")
            cv2.imwrite(amsrcr_output_path, img_amsrcr)

            # MSRCP result
            msrcp_output_path = os.path.join(output_paths['msrcp'], f"{name_without_ext}_msrcp.jpg")
            cv2.imwrite(msrcp_output_path, img_msrcp)

            print(f"Saved all processing results for {img_name}")

        except Exception as e:
            print(f"Error processing image {img_name}: {str(e)}")
            continue


if __name__ == "__main__":
    # Set default paths
    input_folder = 'input'
    output_folder = 'opt_retinex_output'

    # Get paths from command line arguments
    if len(sys.argv) >= 2:
        input_folder = sys.argv[1]
    if len(sys.argv) >= 3:
        output_folder = sys.argv[2]

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    batch_process_images(input_folder, output_folder)
