import os
from LIME import LIME
import cv2
import numpy as np


class DUAL:
    """
    DUAL class for image enhancement, combining forward and reverse illumination
    enhancement and generating final enhanced images through multi-exposure fusion.
    """

    # Initialize LIME operator used in DUAL
    def __init__(self, iterations, alpha, rho, gamma, limestrategy):
        """
        Initialize the core LIME object of DUAL class.

        Parameters:
        - iterations: Number of iterations
        - alpha: Parameter controlling enhancement strength
        - rho: Parameter affecting miu update in optimization process
        - gamma: Parameter adjusting final result brightness
        - limestrategy: LIME strategy used (weight matrix calculation method)
        """
        self.limecore = LIME(iterations, alpha, rho, gamma, limestrategy, exact=True)

    # Load image to be enhanced
    def load(self, imgPath):
        """
        Load image from specified path, normalize and invert it.

        Parameters:
        - imgPath: Image file path
        """
        self.img = cv2.imread(imgPath) / 255  # Read image and normalize to [0, 1]
        self.imgrev = 1 - self.img  # Inverted image for reverse enhancement
        self.imgname = os.path.split(imgPath)[-1]  # Get image filename

    # Multi-exposure image fusion
    def multi_exposureimageFushion(self):
        """
        Fuse original image, forward enhanced image and reverse enhanced image
        using OpenCV's Mertens fusion algorithm.

        Returns:
        - img: Fused image
        """
        mergecore = cv2.createMergeMertens(1, 1, 1)
        img = mergecore.process([self.img, self.forwardimg, self.reverseimg])
        return img

    # Execute DUAL enhancement process
    def run(self):
        """
        Execute complete DUAL image enhancement process:
        1. Forward illumination enhancement
        2. Reverse illumination enhancement
        3. Multi-exposure image fusion to generate final result
        """

        # Step 1: Forward illumination enhancement
        print('Using LIME for forward illumination enhancement!')
        self.limecore.loadimage(self.img)  # Load original image
        self.forwardimg = self.limecore.run()  # Run LIME algorithm to get forward enhanced image
        cv2.imwrite("./pics/DUAL_forward_{}".format(self.imgname), self.forwardimg)  # Save forward enhanced image

        # Step 2: Reverse illumination enhancement
        print('Using LIME for reverse illumination enhancement!')
        self.limecore.loadimage(self.imgrev)  # Load inverted image
        self.reverseimg = 255 - self.limecore.run()  # Get reverse enhanced image and invert back
        cv2.imwrite("./pics/DUAL_reverse_{}".format(self.imgname), self.reverseimg)  # Save reverse enhanced image

        # Step 3: Multi-exposure image fusion
        print('Use multi-exposure image fusion to generate the result!')
        l = self.multi_exposureimageFushion()  # Fuse three images
        cv2.imwrite("./pics/DUAL_result_{}".format(self.imgname), l * 255)  # Save final fusion result

        return l


def build_gaussian_pyramid(img, levels):
    """
    Build Gaussian pyramid

    Parameters:
    - img: Input image
    - levels: Number of pyramid levels

    Returns:
    - pyramid: Gaussian pyramid list, from original size to minimum size
    """
    pyramid = [img]
    current = img.copy()

    for i in range(levels - 1):
        # Use pyrDown for downsampling and smoothing
        current = cv2.pyrDown(current)
        pyramid.append(current)

    return pyramid


def pyramid_dual_enhancement(img_path, output_path, levels=3, iterations=10, alpha=0.1,
                             rho=1.1, gamma=0.6, limestrategy=1):
    """
    DUAL image enhancement based on Gaussian pyramid

    Parameters:
    - img_path: Input image path
    - output_path: Output image path
    - levels: Number of pyramid levels
    - iterations: Number of iterations
    - alpha: Parameter controlling enhancement strength
    - rho: Parameter affecting miu update in optimization process
    - gamma: Parameter adjusting final result brightness
    - limestrategy: LIME strategy used
    """
    # Read image
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Build Gaussian pyramid
    print(f"Building Gaussian pyramid with {levels} levels")
    pyramid = build_gaussian_pyramid(original_img, levels)

    # Enhance each level
    enhanced_pyramid = []
    for i, layer in enumerate(pyramid):
        print(f"Enhancing pyramid level {i + 1}/{levels}")
        # Create temporary file paths
        temp_layer_path = f"temp_pyramid_layer_{i}.png"
        temp_enhanced_path = f"temp_enhanced_layer_{i}.png"

        # Save current layer image
        cv2.imwrite(temp_layer_path, layer)

        try:
            # Use DUAL to enhance each layer
            dual = DUAL(iterations, alpha, rho, gamma, limestrategy)
            dual.load(temp_layer_path)
            enhanced_layer = dual.run()

            # Save enhanced image
            cv2.imwrite(temp_enhanced_path, enhanced_layer * 255)
            enhanced_pyramid.append(temp_enhanced_path)

        except Exception as e:
            print(f"Error enhancing level {i}: {e}")
            # Use original layer when error occurs
            enhanced_pyramid.append(temp_layer_path)
        finally:
            # Clean up temporary files
            if os.path.exists(temp_layer_path):
                os.remove(temp_layer_path)

    # Reconstruct image
    print("Reconstructing enhanced image from pyramid")
    result = reconstruct_from_gaussian_pyramid(enhanced_pyramid)
    cv2.imwrite(output_path, result)
    print(f"Saved enhanced image to {output_path}")

    # Clean up temporary files for enhanced layers
    for path in enhanced_pyramid:
        if os.path.exists(path) and path.startswith("temp_enhanced"):
            os.remove(path)

    return result


def reconstruct_from_gaussian_pyramid(enhanced_paths):
    """
    Reconstruct image from enhanced Gaussian pyramid

    Parameters:
    - enhanced_paths: List of file paths for enhanced pyramid levels

    Returns:
    - reconstructed: Reconstructed image
    """
    # Start reconstruction from bottom level
    layers = [cv2.imread(path) for path in enhanced_paths]
    current = layers[-1]  # Minimum level

    # Upsample and merge layer by layer from bottom to top
    for i in range(len(layers) - 2, -1, -1):
        # Upsample to previous level's size
        current = cv2.pyrUp(current, dstsize=(layers[i].shape[1], layers[i].shape[0]))

        # Use weighted average here
        weight = 0.1  # Adjustable parameter
        current = cv2.addWeighted(current, weight, layers[i], 1 - weight, 0)

    return current


def process_single_image_with_pyramid(img_path, output_path, levels=3, iterations=10,
                                      alpha=0.1, rho=1.1, gamma=0.6, limestrategy=1):
    """
    Image enhancement processing using Gaussian pyramid (with border padding)

    Parameters:
    - img_path: Input image path
    - output_path: Output image path
    - levels: Number of pyramid levels
    - iterations: Number of iterations
    - alpha: Parameter controlling enhancement strength
    - rho: Parameter affecting miu update in optimization process
    - gamma: Parameter adjusting final result brightness
    - limestrategy: LIME strategy used
    """
    # Read original image
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    h, w = original_img.shape[:2]

    # Add border (mirror padding)
    border_size = 30
    padded_img = cv2.copyMakeBorder(
        original_img,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_REFLECT
    )

    # Save temporary image path with border
    temp_dir = os.path.dirname(output_path) or "."
    temp_img_path = os.path.join(temp_dir, f"temp_padded_{os.path.basename(img_path)}")
    cv2.imwrite(temp_img_path, padded_img)

    try:
        # Process with pyramid DUAL
        enhanced_padded_img = pyramid_dual_enhancement(
            temp_img_path, "temp_pyramid_result.png", levels,
            iterations, alpha, rho, gamma, limestrategy
        )

        # Crop out the border
        cropped_img = enhanced_padded_img[
                      border_size:border_size + h,
                      border_size:border_size + w
                      ]

        # Save result
        cv2.imwrite(output_path, cropped_img)
        print(f"Saved result to {output_path}")

    finally:
        # Clean up temporary files
        temp_files = [temp_img_path, "temp_pyramid_result.png"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


def process_images_with_pyramid(input_folder, output_folder, levels=3, iterations=10,
                                alpha=0.1, rho=1.1, gamma=0.6, limestrategy=1):
    """
    Batch process all images in folder (using Gaussian pyramid)

    Parameters:
    - input_folder: Input image folder path
    - output_folder: Output image folder path
    - levels: Number of pyramid levels
    - iterations: Number of iterations
    - alpha: Parameter controlling enhancement strength
    - rho: Parameter affecting miu update in optimization process
    - gamma: Parameter adjusting final result brightness
    - limestrategy: LIME strategy used
    """
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            # Build complete file path
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"PYRAMID_DUAL_result_{filename}")

            try:
                print(f"Processing {filename} with Gaussian pyramid...")
                # Process image with pyramid method
                process_single_image_with_pyramid(
                    input_path, output_path, levels,
                    iterations, alpha, rho, gamma, limestrategy
                )

            except Exception as e:
                print(f"Error processing {filename}: {e}")


def main():
    input_folder = "E:\Desktop\LR2HR\ACCEData"
    output_folder = "E:\Desktop\LR2HR\ACCEData\output"

    # Batch process images (using Gaussian pyramid)
    process_images_with_pyramid(input_folder, output_folder, levels=3,
                                iterations=10, alpha=0.1, rho=1.1, gamma=0.6, limestrategy=1)


if __name__ == "__main__":
    main()
