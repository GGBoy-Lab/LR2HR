import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def histogram_equalization(img_path, output_path):
    """
    Adaptive Histogram Equalization (CLAHE) for image enhancement

    Parameters:
    - img_path: Input image path
    - output_path: Output image path
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Convert to LAB color space for color images
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # Use stronger CLAHE parameters
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        # Convert back to BGR color space
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Process grayscale images directly
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        result = clahe.apply(img)

    cv2.imwrite(output_path, result)
    print(f"Saved histogram equalized image to {output_path}")
    return result


def homomorphic_filtering(img_path, output_path, d0=100, r_l=0.1, r_h=3.0, cutoff=0.5):
    """
    Image enhancement using improved homomorphic filtering

    Parameters:
    - img_path: Input image path
    - output_path: Output image path
    - d0: Gaussian high-pass filter cutoff frequency
    - r_l: Low-frequency gain (controls illumination component)
    - r_h: High-frequency gain (controls reflectance component)
    - cutoff: Cutoff slope of Gaussian filter
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Convert to float and normalize
    img_float = img.astype(np.float32) / 255.0

    # Convert to logarithmic domain
    log_img = np.log(img_float + 1e-8)

    # Get image dimensions
    h, w = log_img.shape[:2]

    # Create homomorphic filter function
    def create_homomorphic_filter(h, w, d0, r_l, r_h, cutoff):
        # Create coordinate grid
        u = np.arange(h).reshape(-1, 1) - h // 2
        v = np.arange(w).reshape(1, -1) - w // 2

        # Calculate distance
        D = np.sqrt(u ** 2 + v ** 2)

        # Gaussian high-pass filter (improved version)
        H_hp = (r_h - r_l) * (1 - np.exp(-cutoff * (D ** 2) / (d0 ** 2))) + r_l

        return H_hp

    # Generate filter
    H = create_homomorphic_filter(h, w, d0, r_l, r_h, cutoff)

    # Process each channel separately
    result = np.zeros_like(log_img)

    if len(log_img.shape) == 3:  # Color images
        for i in range(3):
            # FFT
            fft_img = np.fft.fft2(log_img[:, :, i])
            fft_shift = np.fft.fftshift(fft_img)

            # Apply filter
            filtered_fft = fft_shift * H

            # IFFT
            filtered_ifft = np.fft.ifftshift(filtered_fft)
            filtered_img = np.real(np.fft.ifft2(filtered_ifft))

            result[:, :, i] = filtered_img
    else:  # Grayscale images
        # FFT
        fft_img = np.fft.fft2(log_img)
        fft_shift = np.fft.fftshift(fft_img)

        # Apply filter
        filtered_fft = fft_shift * H

        # IFFT
        filtered_ifft = np.fft.ifftshift(filtered_fft)
        result = np.real(np.fft.ifft2(filtered_ifft))

    # Convert back from logarithmic domain
    result = np.exp(result)

    # Enhance contrast and normalize to [0, 255]
    result = np.clip(result, 0, np.inf)  # Prevent negative values
    for i in range(result.shape[-1] if len(result.shape) > 2 else 1):
        if len(result.shape) > 2:
            min_val = np.min(result[:, :, i])
            max_val = np.max(result[:, :, i])
            if max_val != min_val:  # Prevent division by zero
                result[:, :, i] = ((result[:, :, i] - min_val) / (max_val - min_val)) * 255
        else:
            min_val = np.min(result)
            max_val = np.max(result)
            if max_val != min_val:
                result = ((result - min_val) / (max_val - min_val)) * 255

    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Saved homomorphic filtered image to {output_path}")
    return result


def gamma_correction(img_path, output_path, gamma=0.3):
    """
    Gamma correction for image enhancement

    Parameters:
    - img_path: Input image path
    - output_path: Output image path
    - gamma: Gamma value, greater than 1 darkens, less than 1 brightens, smaller is brighter
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Normalize to [0, 1]
    img_norm = img.astype(np.float32) / 255.0

    # Strong Gamma correction
    img_gamma = np.power(img_norm, 1.0 / gamma)

    # Normalize to [0, 255]
    result = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Saved gamma corrected image to {output_path}")
    return result


def single_scale_retinex(img_path, output_path, sigma=300, restore_factor=1.5, color_gain=8.0):
    """
    Improved Single Scale Retinex (SSR) algorithm for illumination compensation

    Parameters:
    - img_path: Input image path
    - output_path: Output image path
    - sigma: Standard deviation of Gaussian kernel
    - restore_factor: Restoration factor
    - color_gain: Color gain
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Convert to float type
    img_float = img.astype(np.float32) + 1e-8

    # Calculate logarithmic domain image
    log_img = np.log(img_float)

    # Calculate Gaussian blur
    if len(img.shape) == 3:
        # Apply Gaussian blur to each channel separately
        log_blur = np.zeros_like(log_img)
        for i in range(3):
            log_blur[:, :, i] = gaussian_filter(log_img[:, :, i], sigma=sigma)
    else:
        log_blur = gaussian_filter(log_img, sigma=sigma)

    # SSR formula: R(x,y) = log(I(x,y)) - log(L(x,y))
    retinex = log_img - log_blur

    # Apply restoration factor and color gain
    retinex = restore_factor * retinex
    retinex = color_gain * retinex

    # Normalize results to [0, 255]
    for i in range(3 if len(img.shape) == 3 else 1):
        if len(img.shape) == 3:
            min_val = np.min(retinex[:, :, i])
            max_val = np.max(retinex[:, :, i])
            if max_val != min_val:
                retinex[:, :, i] = ((retinex[:, :, i] - min_val) / (max_val - min_val)) * 255
        else:
            min_val = np.min(retinex)
            max_val = np.max(retinex)
            if max_val != min_val:
                retinex = ((retinex - min_val) / (max_val - min_val)) * 255

    result = np.clip(retinex, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Saved Retinex enhanced image to {output_path}")
    return result


def multi_scale_retinex(img_path, output_path, sigmas=[15, 80, 250], weights=None):
    """
    Improved Multi-Scale Retinex (MSR) algorithm

    Parameters:
    - img_path: Input image path
    - output_path: Output image path
    - sigmas: List of standard deviations for different scales
    - weights: Weights for each scale, default is equal weights
    """
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Convert to float type
    img_float = img.astype(np.float32) + 1e-8

    # Calculate logarithmic domain image
    log_img = np.log(img_float)

    # Initialize result array
    retinex = np.zeros_like(log_img)

    # Process each scale
    for i, sigma in enumerate(sigmas):
        if len(img.shape) == 3:
            # Apply Gaussian blur to each channel separately
            log_blur = np.zeros_like(log_img)
            for c in range(3):
                log_blur[:, :, c] = gaussian_filter(log_img[:, :, c], sigma=sigma)
        else:
            log_blur = gaussian_filter(log_img, sigma=sigma)

        # Accumulate weighted results at different scales
        retinex += weights[i] * (log_img - log_blur)

    # Color restoration for the entire image
    retinex = 128 * (1 + retinex)

    # Normalize results to [0, 255]
    for c in range(3 if len(img.shape) == 3 else 1):
        if len(img.shape) == 3:
            min_val = np.min(retinex[:, :, c])
            max_val = np.max(retinex[:, :, c])
            if max_val != min_val:
                retinex[:, :, c] = ((retinex[:, :, c] - min_val) / (max_val - min_val)) * 255
        else:
            min_val = np.min(retinex)
            max_val = np.max(retinex)
            if max_val != min_val:
                retinex = ((retinex - min_val) / (max_val - min_val)) * 255

    result = np.clip(retinex, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Saved Multi-Scale Retinex enhanced image to {output_path}")
    return result


def enhance_contrast_brightness(img_path, output_path, alpha=2.2, beta=40):
    """
    Simple contrast and brightness enhancement

    Parameters:
    - img_path: Input image path
    - output_path: Output image path
    - alpha: Contrast control (1.0-3.0)
    - beta: Brightness control (0-100)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Apply contrast and brightness adjustment
    result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    cv2.imwrite(output_path, result)
    print(f"Saved contrast/brightness enhanced image to {output_path}")
    return result


def bilateral_filter_enhancement(img_path, output_path, d=9, sigma_color=75, sigma_space=75):
    """
    Image detail enhancement using bilateral filtering

    Parameters:
    - img_path: Input image path
    - output_path: Output image path
    - d: Diameter of pixel neighborhood
    - sigma_color: Sigma value in color space
    - sigma_space: Sigma value in coordinate space
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Separate brightness and details
    blur = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    detail = img.astype(np.float32) - blur.astype(np.float32)

    # Enhance details
    enhanced_detail = detail * 2.0  # Enhance detail part

    # Merge results
    result = blur.astype(np.float32) + enhanced_detail
    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Saved bilateral filter enhanced image to {output_path}")
    return result


def unsharp_masking(img_path, output_path, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    """
    Sharpening mask for image enhancement

    Parameters:
    - img_path: Input image path
    - output_path: Output image path
    - kernel_size: Gaussian kernel size
    - sigma: Standard deviation of Gaussian kernel
    - amount: Sharpening intensity
    - threshold: Threshold, changes below this value will not be enhanced
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Convert to float
    img_float = img.astype(float)

    # Create blurred version
    blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)

    # Create mask (detail part)
    mask = img_float - blurred

    # Apply sharpening intensity
    sharpened = img_float + mask * amount

    # Limit pixel value range
    sharpened = np.clip(sharpened, 0, 255)

    # Convert back to uint8
    result = sharpened.astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Saved unsharp masking enhanced image to {output_path}")
    return result


def adaptive_histogram_equalization_advanced(img_path, output_path, clip_limit=4.0, tile_grid_size=(16, 16)):
    """
    Improved Adaptive Histogram Equalization

    Parameters:
    - img_path: Input image path
    - output_path: Output image path
    - clip_limit: Contrast limit
    - tile_grid_size: Grid size
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Convert to LAB color space for color images
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # Use stronger CLAHE parameters
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        # Convert back to BGR color space
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Process grayscale images directly
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        result = clahe.apply(img)

    cv2.imwrite(output_path, result)
    print(f"Saved advanced CLAHE enhanced image to {output_path}")
    return result


def batch_process_images(input_folder, output_base_folder):
    """
    Batch process all images in folder, save results of different methods to different subfolders

    Parameters:
    - input_folder: Input image folder path
    - output_base_folder: Output base folder path
    """
    # Define output folders for different methods
    method_folders = {
        'advanced_clahe': os.path.join(output_base_folder, 'ADVANCED_CLAHE'),
        'homomorphic': os.path.join(output_base_folder, 'HOMOMORPHIC'),
        'gamma': os.path.join(output_base_folder, 'GAMMA'),
        'ssr': os.path.join(output_base_folder, 'SSR'),
        'msr': os.path.join(output_base_folder, 'MSR'),
        'contrast': os.path.join(output_base_folder, 'CONTRAST_BRIGHTNESS'),
        'bilateral': os.path.join(output_base_folder, 'BILATERAL_FILTER'),
        'unsharp': os.path.join(output_base_folder, 'UNSHARP_MASKING')
    }

    # Create output folders
    for folder_path in method_folders.values():
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Get supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            # Build complete file path
            input_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]

            print(f"Processing {filename}...")

            try:
                # 1. Improved Adaptive Histogram Equalization
                hist_eq_path = os.path.join(method_folders['advanced_clahe'], f"{base_name}.png")
                adaptive_histogram_equalization_advanced(input_path, hist_eq_path, clip_limit=4.0,
                                                         tile_grid_size=(16, 16))

                # 2. Improved Homomorphic Filtering
                homomorphic_path = os.path.join(method_folders['homomorphic'], f"{base_name}.png")
                homomorphic_filtering(input_path, homomorphic_path, d0=100, r_l=0.1, r_h=3.0, cutoff=0.5)

                # 3. Stronger Gamma Transformation
                gamma_path = os.path.join(method_folders['gamma'], f"{base_name}.png")
                gamma_correction(input_path, gamma_path, gamma=0.3)

                # 4. Improved Single Scale Retinex
                ssr_path = os.path.join(method_folders['ssr'], f"{base_name}.png")
                single_scale_retinex(input_path, ssr_path, sigma=300, restore_factor=1.5, color_gain=8.0)

                # 5. Improved Multi-Scale Retinex
                msr_path = os.path.join(method_folders['msr'], f"{base_name}.png")
                multi_scale_retinex(input_path, msr_path, sigmas=[15, 80, 250])

                # 6. Contrast and Brightness Enhancement
                contrast_path = os.path.join(method_folders['contrast'], f"{base_name}.png")
                enhance_contrast_brightness(input_path, contrast_path, alpha=2.2, beta=40)

                # 7. Bilateral Filter Enhancement
                bilateral_path = os.path.join(method_folders['bilateral'], f"{base_name}.png")
                bilateral_filter_enhancement(input_path, bilateral_path)

                # 8. Sharpening Mask Enhancement
                unsharp_path = os.path.join(method_folders['unsharp'], f"{base_name}.png")
                unsharp_masking(input_path, unsharp_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")


def compare_methods_side_by_side(image_path):
    """
    Display results of various methods side by side for comparison

    Parameters:
    - image_path: Input image path
    """
    # Read original image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Apply various enhancement methods
    original_result = cv2.imread(image_path)
    clahe_result = adaptive_histogram_equalization_advanced(image_path, "temp_clahe.png", clip_limit=4.0,
                                                            tile_grid_size=(16, 16))
    homomorphic_result = homomorphic_filtering(image_path, "temp_homomorphic.png", d0=100, r_l=0.1, r_h=3.0, cutoff=0.5)
    gamma_result = gamma_correction(image_path, "temp_gamma.png", gamma=0.3)
    ssr_result = single_scale_retinex(image_path, "temp_ssr.png", sigma=300, restore_factor=1.5, color_gain=8.0)
    msr_result = multi_scale_retinex(image_path, "temp_msr.png", sigmas=[15, 80, 250])
    contrast_result = enhance_contrast_brightness(image_path, "temp_contrast.png", alpha=2.2, beta=40)
    bilateral_result = bilateral_filter_enhancement(image_path, "temp_bilateral.png")
    unsharp_result = unsharp_masking(image_path, "temp_unsharp.png")

    # Convert to RGB for matplotlib display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    clahe_rgb = cv2.cvtColor(clahe_result, cv2.COLOR_BGR2RGB)
    homomorphic_rgb = cv2.cvtColor(homomorphic_result, cv2.COLOR_BGR2RGB)
    gamma_rgb = cv2.cvtColor(gamma_result, cv2.COLOR_BGR2RGB)
    ssr_rgb = cv2.cvtColor(ssr_result, cv2.COLOR_BGR2RGB)
    msr_rgb = cv2.cvtColor(msr_result, cv2.COLOR_BGR2RGB)
    contrast_rgb = cv2.cvtColor(contrast_result, cv2.COLOR_BGR2RGB)
    bilateral_rgb = cv2.cvtColor(bilateral_result, cv2.COLOR_BGR2RGB)
    unsharp_rgb = cv2.cvtColor(unsharp_result, cv2.COLOR_BGR2RGB)

    # Display results
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(clahe_rgb)
    axes[0, 1].set_title('Advanced CLAHE')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(homomorphic_rgb)
    axes[0, 2].set_title('Homomorphic Filter')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(gamma_rgb)
    axes[1, 0].set_title('Gamma Correction (γ=0.3)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(ssr_rgb)
    axes[1, 1].set_title('Improved SSR')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(msr_rgb)
    axes[1, 2].set_title('Improved MSR')
    axes[1, 2].axis('off')

    axes[2, 0].imshow(contrast_rgb)
    axes[2, 0].set_title('Contrast & Brightness')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(bilateral_rgb)
    axes[2, 1].set_title('Bilateral Filter')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(unsharp_rgb)
    axes[2, 2].set_title('Unsharp Masking')
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # Delete temporary files
    temp_files = ["temp_clahe.png", "temp_homomorphic.png", "temp_gamma.png",
                  "temp_ssr.png", "temp_msr.png", "temp_contrast.png",
                  "temp_bilateral.png", "temp_unsharp.png"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    # Set input and output folders
    input_folder = "./input"  # Replace with your input image folder path
    output_base_folder = "./enhanced_output"  # Replace with your output base folder path

    # Batch process images
    batch_process_images(input_folder, output_base_folder)

    # Or visualize comparison for a single image
    # compare_methods_side_by_side("./input/test_image.jpg")


if __name__ == "__main__":
    main()
