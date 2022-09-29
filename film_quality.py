"""
Analyze the quality of a sulfide film using lightbox technique.

Created on 08/01/2022

@author: Bryce Smith (brycegsmith@hotmail.com)
"""

from pathlib import Path
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.filters import gaussian
from matplotlib import pyplot as plt


def main():
    """
    Main Function.

    Parameters:
        None

    Returns:
        None

    """

    # Must be [0, 1]. Increase value to only caputure brighter pixels.
    THRESHOLD = 0.6

    print("Started film quality analysis.")
    img_list = import_film_images()
    create_output_directory()
    film_image_analysis(img_list, THRESHOLD)
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

    # Iterate through jpg files in directory
    i = 0
    img_list = []
    for f in p.glob("*.jpg"):
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


def film_image_analysis(img_list, threshold) -> None:
    """
    Analyze each film image. Creates grayscale image, thresholded image, and pixel
    brightness histograms.

    Parameters:
        img_list (list of Paths): list of JPG image filepaths
        threshold (float): value 0 (black) to 1 (white) to threshold image

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
        # Apply gaussian filter to de-noise histogram
        img_gray = imread(img, as_gray=True)
        img_gray = gaussian(img_gray, sigma=3)
        gray_imgs.append(img_gray)
        imsave(p / (title + "_grayscale.png"), img_as_ubyte(img_gray))

        # Create thresholded binary image
        fig = plt.figure()
        binary = img_gray > threshold
        imsave(p / (title + "_theshold.png"), img_as_ubyte(binary))

        # Create brightness histogram
        fig = plt.figure()
        plt.hist(x=img_gray.ravel(), bins=256, range=[0, 1], density=True, alpha=0.5)
        plt.axvline(threshold, color="k", linestyle="dashed", linewidth=1)
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

    plt.title("Normalized Frequency vs Brightness")
    plt.xlabel("Brightness")
    plt.ylabel("Normalized Frequency")
    plt.yticks([], [])
    plt.xlim([0, 1])
    fig.savefig(p / "combined_histogram.png")

    print("Outputted combined histogram.")


if __name__ == "__main__":
    main()
