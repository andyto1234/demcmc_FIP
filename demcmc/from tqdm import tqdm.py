from tqdm import tqdm
import numpy as np
import astropy.units as u
from ashmcmc import ashmcmc, interp_emis_temp
import xarray as xr
import netCDF4 as nc

from demcmc import (
    EmissionLine,
    TempBins,
    load_cont_funcs,
    plot_emission_loci,
    predict_dem_emcee,
    ContFuncDiscrete,
)
from demcmc.units import u_temp, u_dem

def dem_output2xr(dem_out: DEMOutput) -> xr.DataArray:
    """
    Convert the output of a DEM calculation into a DataArray
    """
    temp_centers = dem_out.temp_bins.bin_centers
    temp_edges = dem_out.temp_bins.edges

    samplers = np.arange(dem_out.samples.shape[0])
    coords = {"Sampler": samplers, "Temp bin center": temp_centers.to_value(u_temp)}

    da = xr.DataArray(
        data=dem_out.samples,
        coords=coords,
        attrs={"Temp bin edges": temp_edges.to_value(u_temp)},
    )
    return da

def calc_chi2(mcmc_lines, dem_result, temp_bins):
    chi2 = 0
    for num,i in enumerate(mcmc_lines):
        int_obs = mcmc_lines[num].intensity_obs
        int_pred = mcmc_lines[num]._I_pred(temp_bins, dem_result)
        chi2_line = ((int_pred-int_obs)/mcmc_lines[num].sigma_intensity_obs)**2
        chi2 += chi2_line
    return chi2

def calc_exp(mcmc_lines):
    exp = []
    for num,i in enumerate(mcmc_lines):
        print(f'--------------{num}:{mcmc_lines[num].name}--------------')
        int_obs = mcmc_lines[num].intensity_obs
        int_pred = mcmc_lines[num]._I_pred(temp_bins, dem_median)
        
    print('\n',chi2)

filename = 'SO_EIS_data/eis_20230328_125814.data.h5'
a = ashmcmc(filename)
Lines, Intensity, Int_error = a.fit_data(plot=False)
ldens = a.read_density()

for xpix in tqdm(range(Intensity.shape[1])):
    output_file = filename.split('/')[-1].replace('.data.h5', '') / f"dem_{xpix}.nc"
    # List to save coords that were processed
    ycoords_out = []
    dem_results = []
    chi2_results = []
    linenames = []

    for ypix in range(Intensity.shape[0]):
        logt, emis, linenames = a.read_emissivity(ldens[ypix,xpix])
        logt_interp = interp_emis_temp(logt.value)
        loc = np.where((np.log10(logt_interp)>=4) & (np.log10(logt_interp)<=8))
        # loc = np.where((np.log10(logt.value)>=4.5) & (np.log10(logt.value)<=7.5))
        logt_interp = logt_interp[loc]*u.K
        emis_sorted = a.emis_filter(emis, linenames, Lines)
        temp_bins = TempBins(logt_interp)
        mcmc_lines = []
        for ind, line in tqdm(enumerate(Lines)):
            if Intensity[ypix,xpix,ind] >10:
                mcmc_emis = emis_sorted[ind,:]
                mcmc_emis = ContFuncDiscrete(logt_interp, interp_emis_temp(emis_sorted[ind,:])[loc]* u.cm**5 / u.K, name=line)
                # mcmc_emis = ContFuncDiscrete(logt_interp, emis_sorted[ind,:][loc]* u.cm**5 / u.K, name=line)
                mcmc_intensity = Intensity[ypix,xpix,ind]
                if (Int_error[ypix,xpix,ind] > Intensity[ypix,xpix,ind]*0.25) :
                    mcmc_int_error = Int_error[ypix,xpix,ind]
                else:
                    mcmc_int_error = Intensity[ypix,xpix,ind]*0.25
                emissionLine = EmissionLine(
                    mcmc_emis,
                    intensity_obs=mcmc_intensity,
                    sigma_intensity_obs=mcmc_int_error,
                    name=line
                )
                mcmc_lines.append(emissionLine)


        # dem_result = predict_dem_emcee(mcmc_lines, temp_bins, nwalkers=200, nsteps=400, progress=True, dem_guess=None)
        dem_result = predict_dem_emcee(mcmc_lines, temp_bins, nwalkers=200, nsteps=300, progress=False, dem_guess=None)
        dem_init = np.median([sample.values.value for num, sample in enumerate(dem_result.iter_binned_dems())],axis=0)
        dem_result = predict_dem_emcee(mcmc_lines, temp_bins, nwalkers=200, nsteps=1000, progress=False, dem_guess=dem_init)

        dem_median = np.median([sample.values.value for num, sample in enumerate(dem_result.iter_binned_dems())],axis=0)
        dem_results.append(dem_median)
        chi2 = calc_chi2(mcmc_lines, dem_median, temp_bins)
        chi2_results.append(chi2)
        ycoords_out.append(ypix)
        lines_used = np.array(linenames.append(mcmc_lines), dtype=object)

    np.savez(f'{a.outdir}/{xpix:03d}.npz', dem_results=dem_results, chi2=chi2_results, ycoords_out=ycoords_out, lines_used=lines_used)


