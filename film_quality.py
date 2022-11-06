"""
Analyze the quality of a sulfide film using lightbox technique.

Created on 10/01/2022

@author: Bryce Smith
"""

from pathlib import Path
from skimage.io import imread, imsave, imshow
from skimage.util import img_as_ubyte
from skimage.filters import gaussian
from skimage.morphology import opening
from matplotlib import pyplot as plt
from matplotlib import lines
import numpy as np


def main():
    """
    Main Function.

    Parameters:
        None

    Returns:
        None

    """

    # BRYCE PARAMETERS
    # FILM_THRESHOLD = 0.65
    # QUALITY_THRESHOLD = 0.14

    # DAVID PARAMETERS
    print("David")
    FILM_THRESHOLD = 0.71
    QUALITY_THRESHOLD = 0.28

    # Must be [0, 1]. Brightness values BELOW threshold are part of film.
    # FILM_THRESHOLD = 0.71

    # Must be [0, 1]. Brightness values BELOW threshold are high-quality film.
    # QUALITY_THRESHOLD = 0.26

    print("Started film quality analysis.")
    img_list = import_film_images()
    create_output_directory()
    film_image_analysis(img_list, FILM_THRESHOLD, QUALITY_THRESHOLD)
    print("Completed film quality analysis.")


def import_film_images() -> list:
    """
    Get list of JPG image filepaths in local film_images folder.

    Parameters:
        None

    Returns:
        img_list (list of Paths): list of JPG image filepaths
    """

    # Check if film_images directory exists
    p = Path("./film_images")
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        msg = "Did not find film_images directory. Upload images to created directory."
        print(msg)

    # Iterate through jpg files in directory (supports .jpg or .jpeg endings)
    i = 0
    img_list = []
    for f in p.glob("*.jp*g"):
        i += 1
        img_list.append(f)
    msg = "Found " + str(i) + " jpg image(s) in directory. Stop execution if incorrect."
    print(msg)

    return img_list


def create_output_directory() -> None:
    """
    Create a local output directory to store results of image analysis.

    Parameters:
        None

    Returns:
        None
    """

    # Create output directory if does not exist
    p = Path("./output")
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def film_image_analysis(img_list, film_threshold, quality_threshold) -> None:
    """
    Analyze each film image. Creates grayscale image, thresholded image, and pixel
    brightness histograms.

    Parameters:
        img_list (list of Paths): list of JPG image filepaths
        film_threshold (float): value 0 (black) to 1 (white) to threshold film
        quality_threshold (float): value 0 (black) to 1 (white) to threshold quality

    Returns:
        None
    """

    # Filepath for output
    p = Path("./output")

    # Individual outputs
    titles = []
    gray_imgs = []
    for img in img_list:
        # Image title
        title = img.stem
        titles.append(title)

        # Create grayscale image [0 = black, 1 = white]. Save as uint8 [0, 255].
        # Apply gaussian filter to remnove low-dimensional image noise
        img_gray = imread(img, as_gray=True)
        img_gray = gaussian(img_gray, sigma=3)
        gray_imgs.append(img_gray)
        imsave(p / (title + "_grayscale.png"), img_as_ubyte(img_gray))

        # Threshold to capture all film pixels
        # Morphological opening to remove small dots
        film_binary = img_gray < film_threshold
        element = np.ones([10, 10])
        film_binary = opening(film_binary, element)
        film_px = np.sum(film_binary)

        # Threshold to capture high-quality film pixels
        high_quality_binary = img_gray < quality_threshold
        high_quality_px = np.sum(high_quality_binary)

        # Combine thresholds to capture low-quality film pixels
        low_quality_binary = film_binary == np.invert(high_quality_binary)
        low_quality_px = np.sum(low_quality_binary)

        # Create threshold array (black = high-quality film, red = low-quality)
        x = img_gray.shape[0]
        y = img_gray.shape[1]
        thresholded = np.ones((x, y, 3), dtype=np.uint8) * 255
        thresholded[film_binary] = [0, 0, 0]
        thresholded[low_quality_binary] = [255, 0, 0]
        imsave(p / (title + "_theshold.png"), thresholded)

        # Print quality metric
        quality = round(high_quality_px / film_px * 100, 1)
        print("Film Quality Metric: " + str(quality) + "%")

        # Create brightness histogram
        fig = plt.figure()
        bin_count = 256
        hist_range = [0, 1]
        hist_output = plt.hist(
            x=img_gray.ravel(),
            bins=bin_count,
            range=hist_range,
            density=True,
            alpha=0.5,
        )
        plt.axvline(film_threshold, color="k", linestyle="dashed", linewidth=1)
        plt.axvline(quality_threshold, color="r", linestyle="dashed", linewidth=1)
        quality_legend = lines.Line2D(
            [],
            [],
            color="r",
            linestyle="--",
            label="Quality Threshold",
        )
        film_legend = lines.Line2D(
            [],
            [],
            color="k",
            linestyle="--",
            label="Film Threshold",
        )
        plt.legend(handles=[quality_legend, film_legend])
        plt.title("Normalized Frequency vs Brightness")
        plt.xlabel("Brightness")
        plt.ylabel("Normalized Frequency")
        plt.yticks([], [])
        plt.xlim([0, 1])
        fig.savefig(p / (title + "_histogram.png"))

        print("Outputted grayscale/threshold image & histogram for " + title + ".")

    # Combined histogram
    fig = plt.figure()
    for img in gray_imgs:
        plt.hist(x=img.ravel(), bins=256, range=[0, 1], density=True, alpha=0.5)
        plt.legend(titles)
    plt.axvline(film_threshold, color="k", linestyle="dashed", linewidth=1)
    plt.axvline(quality_threshold, color="r", linestyle="dashed", linewidth=1)
    plt.title("Normalized Frequency vs Brightness")
    plt.xlabel("Brightness")
    plt.ylabel("Normalized Frequency")
    plt.yticks([], [])
    plt.xlim([0, 1])
    fig.savefig(p / "combined_histogram.png")

    print("Outputted combined histogram.")


if __name__ == "__main__":
    main()
