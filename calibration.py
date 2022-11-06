"""
Get calibration parameters for camera used for film quality imaging.

Created on 11/01/2022

@author: Bryce Smith
"""

from pathlib import Path
from skimage.io import imread, imsave, imshow
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def main():
    """
    Main Function.

    Parameters:
        None

    Returns:
        None

    """

    print("Started calibration analysis.")
    img_list = import_calibration_images()
    create_output_directory()
    calibration_analysis(img_list)
    print("Completed calibration analysis.")


def import_calibration_images() -> list:
    """
    Get list of JPG image filepaths in local calibration folder.

    Parameters:
        None

    Returns:
        img_list (list of Paths): list of JPG image filepaths
    """

    # Check if film_images directory exists
    p = Path("./calibration_images")
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        msg = "Did not find calibration directory. Upload images to created directory."
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
    Create a local output directory to store results of calibration analysis.

    Parameters:
        None

    Returns:
        None
    """

    # Create output directory if does not exist
    p = Path("./output")
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def calibration_analysis(img_list) -> None:
    """
    Analyze each calibration image. Creates grayscale image, identifies peaks, and
    outputs calibration parameters.

    Parameters:
        img_list (list of Paths): list of JPG image filepaths

    Returns:
        None
    """

    # Filepath for output
    p = Path("./output")

    # Variables to store calibration parameters
    white_peaks = []
    peak_to_peaks = []

    # Individual outputs
    for img in img_list:
        # Image title
        title = img.stem

        # Create grayscale image [0 = black, 1 = white]. Save as uint8 [0, 255].
        img_gray = imread(img, as_gray=True)

        # Create brightness histogram
        bin_count = 256
        hist_range = [0, 1]
        hist_output = plt.hist(
            x=img_gray.ravel(),
            bins=bin_count,
            range=hist_range,
            density=True,
            alpha=0.5,
        )

        # Create trace of brightness histogram
        fig = plt.figure()
        x = hist_output[0]
        peaks, _ = find_peaks(x, height=10.0, width=2.0)
        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        fig.savefig(p / (title + "_calibration.png"))

        # Calculate calibration parameters
        peak_to_peak = peaks[1] - peaks[0]
        peak_to_peaks.append(peak_to_peak)
        print("Peak-To-Peak: " + str(peak_to_peak))

        print("Outputted calibration trace & parameters for " + title + ".")

    # Output average calibration parameters
    peak_to_peak_avg = sum(peak_to_peaks) / len(peak_to_peaks)
    print("Peak-To-Peak: " + str(peak_to_peak_avg))


if __name__ == "__main__":
    main()
