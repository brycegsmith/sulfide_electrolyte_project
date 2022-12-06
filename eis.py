"""
Analysis of EIS data from sulfide electrolyte wet processing DOE.

Created on 11/06/2022

@author: Bryce Smith
"""

from pathlib import Path
from galvani import BioLogic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from impedance.preprocessing import ignoreBelowX
from impedance.models.circuits import CustomCircuit, fitting
from impedance.visualization import plot_nyquist, plot_bode
import seaborn as sns


def main():
    """
    Main Function.

    Parameters:
        None

    Returns:
        None

    """

    print("Started EIS analysis.")

    # Backend functions (DO NOT EDIT)
    files = import_mpr_files()
    datapath = import_datasheet()
    create_output_directory()

    # Constants (DO NOT EDIT)
    FILM_AREA_CM2 = 0.71

    # Fit equivalent circuit to all data w/ option to plot and/or export data
    # CHANGE NAMING CONVENTION FOR NEW DATASETS (SEE FUNCTION)
    """
    eis_analysis(files, datapath, FILM_AREA_CM2, save_data=False, plot=False)
    """

    # Create combined Nyquist plot of files containing all substrings
    """
    nyquist_plot_group(
        files,
        contains_list=["D011"],
        mode="any",
        suppress_legend=False,
        group_title="Nyquist Plot: D011",
    )
    """

    # Create boxplot for analysis of specific processing conditions on conductivity
    """
    x_var = "time_seconds"
    hue_var = None
    filter_dict = {
        "pressure_tons": [0, 1, 2, 3, 4],
        "time_seconds": [0, 120, 280, 440, 600],
    }
    create_boxplot(datapath, x_var, filter_dict, hue_var)
    """

    # Create boxplot of conductivity for all samples to visualize replicate variability
    """
    hue_var = "time_seconds"
    create_sample_boxplot(datapath, hue_var)
    """

    print("Completed EIS analysis.")


def import_mpr_files() -> list:
    """
    Get list of BioLogic mpr files in eis_data folder.

    Parameters:
        None

    Returns:
        files (list of Paths): list of mpr filepaths
    """

    # Check if eis_data directory exists
    p = Path("./eis_data")
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        msg = "Did not find eis_data directory. Upload data to created directory."
        print(msg)
        return

    # Iterate through mpr files in directory
    i = 0
    files = []
    for f in p.glob("*.mpr"):
        i += 1
        files.append(f)
    msg = "Found " + str(i) + " mpr file(s) in directory. Stop execution if incorrect."
    print(msg)

    return files


def import_datasheet() -> Path:
    """
    Get filepath of datasheet.csv sheet for writing & reading metadata.

    Parameters:
        None

    Returns:
        datapath (Path): CSV with film metadata
    """

    # Check if datasheet directory exists
    p = Path("./datasheet")
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        msg = "Did not find datasheet directory. Upload data to created directory."
        print(msg)
        return

    # Find correct csv with metadata
    if len(list(p.glob("datasheet.csv"))) == 0:
        msg = "Did not find datasheet.csv in datasheet directory. Please upload."
        print(msg)
        return
    else:
        datapath = next(p.glob("datasheet.csv"))
        msg = "Imported datasheet file."
        print(msg)

    return datapath


def create_output_directory() -> None:
    """
    Create a local output directory to store results of analysis.

    Parameters:
        None

    Returns:
        None
    """

    # Create output directory if does not exist
    p = Path("./output")
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def eis_fitting(mpr_file, print_circuit=False) -> list:
    """
    Perform fitting of equivalent circuit on EIS data.

    Parameters:
        mpr_file (list of Paths): mpr filepath
        print_circuit (boolean): flag indicating whether to print all circuit parameters

    Returns:
        fitting_output (list): [circuit, frequencies, Z, Z_fit]
    """

    # Convert mpr file to dataframe
    mpr = BioLogic.MPRfile(str(mpr_file))
    df = pd.DataFrame(mpr.data)

    # Convert to numpy arrays and reverse (low to high freq) for impedance library
    freq = np.flip(df["freq/Hz"].to_numpy())
    real = np.flip(df["Re(Z)/Ohm"].to_numpy())
    imag = np.flip(df["-Im(Z)/Ohm"].to_numpy())

    # Convert to format for impedance library and trim below x-axis
    Z = (real + 1j * -1.0 * imag).astype(np.complex128)
    freq, Z = ignoreBelowX(freq, Z)

    # Estimate RQ-element's resistor impedance by local minima of Bode plot
    # Local minima is found by ABSOLUTE minimum of plot due to dataset
    phase_angle = -np.angle(Z, deg=True)
    idx = np.argmin(phase_angle)
    R1 = float(real[idx])

    # Estimate RQ-element's CPE Q value as 10^-6
    # More sophisticated estimation methods exist but this works well on dataset
    Q1 = 10 ** -6
    Q2 = 10 ** -6

    # Estimate both CPE alpha values [0, 1] as 1.0
    a1 = 1.0
    a2 = 1.0

    # Create and fit ciruit to experimental data
    circuit = "p(R1,CPE1)-CPE2"
    initial_guess = [R1, Q1, a1, Q2, a2]
    circuit = CustomCircuit(circuit, initial_guess=initial_guess)
    circuit.fit(freq, Z)
    Z_fit = circuit.predict(freq)

    # Print all circuit parameters if desired
    if print_circuit:
        print(circuit)

    return [circuit, freq, Z, Z_fit]


def eis_analysis(files, datapath, film_area_cm2, save_data=False, plot=False) -> None:
    """
    Equivalent circuit fitting for all files. Option to create Nyquist/Bode plots and
    export data as csv.

    NAMING CONVENTION: Wet Process Optimization DOE
        EIS files names must follow naming convention *_TX_PX_YX_CXX
            * (Any Alphanumeric): Slurry Sample ID
            TX: Casting Thickness ID
            PX: Pressing ID
            YX: Replicate ID
            CXX: EIS Channel ID

    NAMING CONVENTION: Pressing DOE
        EIS files names must follow naming convention PXTXYX_CXX
            PX: Pressing Pressure (tons)
            TX: Pressing Time (rounded minutes)
            YX: Replicate ID
            CXX: EIS Channel ID

    Parameters:
        files (list of Paths): list of mpr filepaths
        datapath (Path): CSV with film metadata
        film_area_cm2 (float): film area in cm2
        save_data (Boolean): flag whether to save extracted values to datasheet CSV
        plot (Boolean): flag whether to plot impedance data

    Returns:
        None
    """

    # Dataframe to export extracted data
    datasheet = pd.read_csv(datapath)

    # Must have "sample_id" column that matches MPR file naming convention
    datasheet.set_index("sample_id", inplace=True)

    for f in files:
        # Perform fitting on EIS data and extract circuit
        print("Fitting equivalent circuit: " + str(f))
        fitting_output = eis_fitting(f)
        circuit = fitting_output[0]

        # Extract circuit R1 (total) resistance
        idx = circuit.get_param_names()[0].index("R1")
        R1_fit = circuit.parameters_[idx]
        print("R1 Resistance: " + str(round(R1_fit, 3)) + " Ohms")

        # NAMING CONVENTION: Wet Process Optimization DOE (EDIT)
        """
        split_str = f.parts[-1].split("_Y")
        sample_id = split_str[0]
        replicate_id = int(split_str[1].split("_C")[0])
        """

        # NAMING CONVENTION: Pressing DOE (EDIT)
        split_str = f.parts[-1].split("Y")
        sample_id = split_str[0]
        replicate_id = int(split_str[1].split("_C")[0])

        # Calculate ionic conductivity (S/cm)
        # Thickness column(s) must be labeled yX_thickness (X = integer)
        t_um = int(datasheet.loc[sample_id, "y" + str(replicate_id) + "_thickness"])
        cond = (t_um / 10 ** 4) / (R1_fit * film_area_cm2)

        # Calculate RMSE between Z and Z_fit
        rmse = fitting.rmse(fitting_output[2], fitting_output[3])

        # Populate resistance, rmse, and conductivity
        if save_data:
            datasheet.loc[sample_id, "y" + str(replicate_id) + "_r1"] = R1_fit
            datasheet.loc[sample_id, "y" + str(replicate_id) + "_rmse_circuit"] = rmse
            datasheet.loc[sample_id, "y" + str(replicate_id) + "_conductivity"] = cond

        # Create individual Nyquist & Bode plots
        if plot:
            label = sample_id + "_Y" + str(replicate_id)
            plot_eis({label: fitting_output}, nyquist=True, bode=False)

    if save_data:
        # Recompute averages
        mask = datasheet.columns.str.contains("y.*_thickness")
        datasheet["thickness_avg"] = datasheet.loc[:, mask].mean(axis=1)

        mask = datasheet.columns.str.contains("y.*_conductivity")
        datasheet["conductivity_avg"] = datasheet.loc[:, mask].mean(axis=1)

        # Recompute standard deviations
        mask = datasheet.columns.str.contains("y.*_thickness")
        datasheet["thickness_std"] = datasheet.loc[:, mask].std(axis=1)

        mask = datasheet.columns.str.contains("y.*_conductivity")
        datasheet["conductivity_std"] = datasheet.loc[:, mask].std(axis=1)

        # Save as CSV
        datasheet.to_csv(datapath)


def plot_eis(sample_dict, nyquist=True, bode=False, **kwargs) -> None:
    """
    Create individual and group Nyquist plots. Create individual Bode plots.

    Parameters:
        sample_dict (dict): dictionary of form label : fitting_output
            label (str): label to identify film
            fitting_output (list): [circuit, frequencies, Z, Z_fit]
        nyquist (boolean): flag whether to create nyquist plot
        bode (boolean): flag whether to create bode plot
        kwargs: allows user to pass arguments plot GROUP NYQUIST PLOT customization
            group_title (str): custom plot title
            suppress_legend (boolean): flag to suppress legend
            show_group_plot (boolean): flag to show plot

    Returns:
        None
    """

    # Filepath for output
    p = Path("./output")

    # Create individual plots
    for label in sample_dict.keys():
        # Extract values from fitting output
        fitting_output = sample_dict.get(label)
        freq = fitting_output[1]
        Z = fitting_output[2]
        Z_fit = fitting_output[3]

        # Create individual Nyquist plot
        if nyquist:
            fig, ax = plt.subplots()
            plot_nyquist(ax, Z, fmt=".")
            plot_nyquist(ax, Z_fit, fmt="-")

            # Nyquist plot formatting
            plt.xlim([0, 2000])
            plt.ylim([0, 2000])
            ax.set_aspect("equal", adjustable="box")
            plt.xlabel("$Z_{re} (Ω)$", fontsize=13)
            plt.ylabel("$-Z_{im} (Ω)$", fontsize=13)
            plt.title("Nyquist Plot: " + label, fontsize=15)
            plt.legend(["Data", "Fit"], loc="lower right")
            fig.savefig(p / (label + "_nyquist.png"))
            plt.close()

        # Create individual Bode plot
        if bode:
            # Catch random formatting error due to issue with impedance library
            fig, ax = plt.subplots(2, 1)
            ax = plot_bode(ax, freq, Z, fmt="s")
            ax = plot_bode(ax, freq, Z_fit, fmt="-")

            # Bode plot formatting
            fig.set_size_inches(9, 7)
            ax[0].set_title("Bode Plot: " + label, fontsize=20)
            ax[0].set_xlabel("")
            plt.savefig(p / (label + "_bode.png"))
            plt.close()

    # Create group Nyquist plot (if applicable)
    if nyquist and len(sample_dict) > 1:
        fig, ax = plt.subplots()
        colormap = plt.cm.viridis(np.linspace(0, 1, len(sample_dict)))

        for label, color in zip(sample_dict.keys(), colormap):
            # Extract values from fitting output
            fitting_output = sample_dict.get(label)
            freq = fitting_output[1]
            Z = fitting_output[2]
            Z_fit = fitting_output[3]

            # Plot raw and fit Nyquist data
            plot_nyquist(ax, Z, fmt=".", c=color, label=label)
            plot_nyquist(ax, Z_fit, fmt="-", c=color)

        # Nyquist plot formatting
        plt.xlim([0, 2000])
        plt.ylim([0, 2000])
        ax.set_aspect("equal", adjustable="box")
        plt.xlabel("$Z_{re} (Ω)$", fontsize=13)
        plt.ylabel("$-Z_{im} (Ω)$", fontsize=13)

        # Custom formatting w/ kwargs
        if "group_title" in kwargs:
            plt.title(kwargs.get("group_title"))
        else:
            plt.title("Nyquist Plot")

        if "suppress_legend" not in kwargs or not kwargs.get("suppress_legend"):
            plt.legend(loc="lower right")

        if "show" in kwargs:
            plt.show()

        # Save and close plot
        fig.savefig(p / "group_nyquist.png")
        plt.close()


def nyquist_plot_group(files, contains_list, mode, **kwargs) -> None:
    """
    Plot a group of EIS data and fit equivalent circuits given list of filename
    substrings.

    Parameters:
        files (list of Paths): list of mpr filepaths
        contains_list (list of strs): substrings in filename by mode below
        mode (str): "all" substrings contained or "any" substrings contained
        kwargs: allows user to pass arguments plot GROUP NYQUIST PLOT customization
            group_title (str): custom plot title
            suppress_legend (boolean): flag to suppress legend
            show_group_plot (boolean): flag to show plot

    Returns:
        None
    """

    # Initialize dictionary to store group labels and extracted fit data
    sample_dict = dict()

    # Iterate through all mpr files for correct subset
    for f in files:
        label = f.parts[-1].split(".mpr")[0]

        if mode == "all":
            if all(_ in label for _ in contains_list):
                sample_dict[label] = eis_fitting(f, print_circuit=False)
        elif mode == "any":
            if any(_ in label for _ in contains_list):
                sample_dict[label] = eis_fitting(f, print_circuit=False)

    # Plot group Nyquist and individual for each label
    plot_eis(sample_dict, **kwargs)


def create_boxplot(datapath, x_var, filter_dict, hue_var=None) -> None:
    """
    Create boxplot to compare processing conditions between films

    Parameters:
        datapath (Path): CSV with film metadata
        x_var (str): variable to be plotted on x-axis of boxplot
        filter_dict (dict): dictionary of form variable_name : [conditions to include]
        hue_var (str or None): variable to color boxs (create 3d plot) (if desired)

    Returns:
        None
    """

    # Filepath for output
    p = Path("./output")

    # Dataframe to export extracted data
    datasheet = pd.read_csv(datapath)

    # DATA FORMATTING FOR WET PROCESSING DATASHEET ONLY
    """
    # Filter dataframe
    mask_a = datasheet["A"].isin(filter_dict.get("A"))
    mask_b = datasheet["B"].isin(filter_dict.get("B"))
    mask_c = datasheet["C"].isin(filter_dict.get("C"))
    mask_d = datasheet["D"].isin(filter_dict.get("D"))
    mask = mask_a & mask_b & mask_c & mask_d
    df = datasheet[mask]

    # Rename x_var for readability
    if x_var == "A":
        x_var = "lpsc_nbr_ratio"
    elif x_var == "B":
        x_var = "solid_liquid_ratio"
    elif x_var == "C":
        x_var = "cast_thickness"
    elif x_var == "D":
        x_var = "press_condition"
    """

    # DATA FORMATTING FOR PRESSING DATASHEET ONLY
    mask_pressure = datasheet["pressure_tons"].isin(filter_dict.get("pressure_tons"))
    mask_time = datasheet["time_seconds"].isin(filter_dict.get("time_seconds"))
    mask = mask_pressure & mask_time
    df = datasheet[mask]

    # Create boxplot of conductivity vs. x_var
    if hue_var is None:
        sns.boxplot(x=x_var, y="conductivity_avg", data=df)
    else:
        # DATA FORMATTING FOR WET PROCESSING DATASHEET ONLY
        """
        if hue_var == "A":
            hue_var = "lpsc_nbr_ratio"
        elif hue_var == "B":
            hue_var = "solid_liquid_ratio"
        elif hue_var == "C":
            hue_var = "cast_thickness"
        elif hue_var == "D":
            hue_var = "press_condition"
        """
        sns.boxplot(x=x_var, y="conductivity_avg", data=df, hue=hue_var)

    # Formatting
    plt.yscale("log")
    plt.ylim(0.00001, 0.001)  # Y-Axis Limits (MAY NEED TO EDIT)
    plt.ylabel("Ionic Conductivity (S/cm)")
    plt.savefig(p / "boxplot.png", bbox_inches="tight")


def create_sample_boxplot(datapath, hue_var=None) -> None:
    """
    Create a boxplot of each sample (replicates) vs. conductivity.

    Parameters:
        datapath (Path): CSV with film metadata
        hue_var (str or None): variable to color boxs (create 3d plot) (if desired)

    Returns:
        None
    """

    # Filepath for output
    p = Path("./output")

    # Dataframe to export extracted data
    datasheet = pd.read_csv(datapath)

    # Filter and reformat only sample_id + replicate columns
    columns = ["y1_conductivity", "y2_conductivity", "y3_conductivity"]
    replicates = pd.melt(datasheet.loc[:, columns])["value"]
    sample_id = datasheet.loc[:, "sample_id"]
    sample_id = pd.concat([sample_id, sample_id, sample_id], axis=0, ignore_index=True)
    df = pd.concat([sample_id, replicates], axis=1)

    # Create boxplot of conductivity vs. sample_id
    if hue_var is None:
        sns.boxplot(x="sample_id", y="value", data=df)
    else:
        hue_column = []
        datasheet.set_index("sample_id", inplace=True)
        for id in sample_id.to_list():
            hue_value = datasheet.at[id, hue_var]
            hue_column.append(hue_value)
        df[hue_var] = hue_column
        sns.boxplot(x="sample_id", y="value", hue=hue_var, dodge=False, data=df)

    # Formatting
    plt.yscale("log")
    plt.ylim(0.00001, 0.001)  # Y-Axis Limits (MAY NEED TO EDIT)
    plt.xticks(rotation=45)
    plt.xlabel("Sample ID")
    plt.ylabel("Ionic Conductivity (S/cm)")
    plt.savefig(p / "sample_boxplot.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
