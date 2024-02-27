from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import numpy as np
import astropy.units as u
from ashmcmc import ashmcmc, interp_emis_temp

from demcmc import (
    EmissionLine,
    TempBins,
    load_cont_funcs,
    plot_emission_loci,
    predict_dem_emcee,
    ContFuncDiscrete,
)
from demcmc.units import u_temp, u_dem

def calc_chi2(mcmc_lines: list[EmissionLine], dem_result: np.array, temp_bins: TempBins) -> float:
    # Calculate the chi-square value for the given MCMC lines, DEM result, and temperature bins
    int_obs = np.array([line.intensity_obs for line in mcmc_lines])
    int_pred = np.array([line._I_pred(temp_bins, dem_result) for line in mcmc_lines])
    sigma_intensity_obs = np.array([line.sigma_intensity_obs for line in mcmc_lines])

    chi2_line = ((int_pred - int_obs) / sigma_intensity_obs) ** 2
    chi2 = np.sum(chi2_line)
    return chi2

def mcmc_process(mcmc_lines: list[EmissionLine], temp_bins: TempBins) -> np.ndarray:
    # Perform MCMC process for the given MCMC lines and temperature bins
    dem_result = predict_dem_emcee(mcmc_lines, temp_bins, nwalkers=200, nsteps=300, progress=False, dem_guess=None)
    dem_init = np.median([sample.values.value for num, sample in enumerate(dem_result.iter_binned_dems())], axis=0)
    dem_result = predict_dem_emcee(mcmc_lines, temp_bins, nwalkers=200, nsteps=1000, progress=False,
                                    dem_guess=dem_init)
    dem_median = np.median([sample.values.value for num, sample in enumerate(dem_result.iter_binned_dems())],
                            axis=0)

    return dem_median

def check_dem_exists(filename: str) -> bool:
    # Check if the DEM file exists
    from os.path import exists
    return exists()    

def process_pixel(args: tuple[int, np.ndarray, np.ndarray, list[str], np.ndarray, ashmcmc]) -> None:
    # Process a single pixel with the given arguments
    xpix, Intensity, Int_error, Lines, ldens, a = args
    output_file = f'{a.outdir}/dem_{xpix}.npz'
    ycoords_out = []
    dem_results = []
    chi2_results = []
    linenames_list = []

    if not check_dem_exists(output_file):
        for ypix in tqdm(range(Intensity.shape[0])):

            logt, emis, linenames = a.read_emissivity(ldens[ypix, xpix])
            logt_interp = interp_emis_temp(logt.value)
            loc = np.where((np.log10(logt_interp) >= 4) & (np.log10(logt_interp) <= 8))
            logt_interp = logt_interp[loc] * u.K
            emis_sorted = a.emis_filter(emis, linenames, Lines)
            temp_bins = TempBins(logt_interp)
            mcmc_lines = []

            for ind, line in enumerate(Lines):
                if Intensity[ypix, xpix, ind] > 10:
                    mcmc_emis = emis_sorted[ind, :]
                    mcmc_emis = ContFuncDiscrete(logt_interp, interp_emis_temp(emis_sorted[ind, :])[loc] * u.cm ** 5 / u.K,
                                                name=line)
                    mcmc_intensity = Intensity[ypix, xpix, ind]
                    mcmc_int_error = max(Int_error[ypix, xpix, ind], 0.25 * mcmc_intensity)
                    emissionLine = EmissionLine(
                        mcmc_emis,
                        intensity_obs=mcmc_intensity,
                        sigma_intensity_obs=mcmc_int_error,
                        name=line
                    )
                    mcmc_lines.append(emissionLine)

            dem_median = mcmc_process(mcmc_lines, temp_bins)
            dem_results.append(dem_median)
            chi2 = calc_chi2(mcmc_lines, dem_median, temp_bins)
            chi2_results.append(chi2)
            ycoords_out.append(ypix)
            linenames_list.append(mcmc_lines)
        dem_results = np.array(dem_results)
        chi2_results = np.array(chi2_results)
        linenames_list = np.array(linenames_list, dtype=object)

        np.savez(output_file, dem_results=dem_results, chi2=chi2_results, ycoords_out=ycoords_out, lines_used=linenames_list, logt = np.array(logt_interp))

def download_data(filename: str) -> None:
    from eispac.download import download_hdf5_data
    download_hdf5_data(filename, local_top='SO_EIS_data', overwrite=False)

def process_data(filename: str) -> None:
    # Create an ashmcmc object with the specified filename
    import platform
    download_data(filename)
    a = ashmcmc(filename)

    # Retrieve necessary data from ashmcmc object
    Lines, Intensity, Int_error = a.fit_data(plot=False)
    ldens = a.read_density()

    # Generate a list of arguments for process_pixel function
    args_list = [(xpix, Intensity, Int_error, Lines, ldens, a) for xpix in range(Intensity.shape[1])]

    # Determine the operating system type (Linux or macOS)
    system_type = platform.system()

    # Set the number of processes based on the operating system
    if system_type == "Linux":
        process_num = 64
    elif system_type == "Darwin":
        process_num = 10
    else:
        process_num = 4  # Default value for other operating systems

    # Create a Pool of processes for parallel execution
    with Pool(processes=process_num) as pool:
        results = list(tqdm(pool.imap(process_pixel, args_list), total=len(args_list), desc="Processing Pixels"))

if __name__ == "__main__":
    filename = 'SO_EIS_data/eis_20230405_220513.data.h5'
    process_data(filename)