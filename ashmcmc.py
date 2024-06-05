from demcmc_FIP.asheis import asheis
from multiprocessing import Pool
import os
import re
import numpy as np
import configparser

def find_matching_file(log_density, abund_file = 'emissivities_sun_photospheric_2015_scott'):
    import platform

    config_obj = configparser.ConfigParser()
    config_obj.read("demcmc_FIP/configfile.ini")
    directories = config_obj["directories"]
    main_dir = directories['main_dir']
    directory = main_dir+abund_file+'/'

#    if platform.system() == 'Linux':
##        directory=f'/home/staff/daithil/work/python_output/FIP/{abund_file}/'
#        directory=f'/disk/solar2/dml/python/FIP/{abund_file}/'
#
#    if platform.system() == 'Darwin':
#        directory=f'/Users/dml/python_output/FIP/{abund_file}/'

    # Convert log_density to float
    target_log_density = float(log_density)
    
    matching_file = None
    min_density_difference = float('inf')

    for filename in os.listdir(directory):
        if filename.startswith("emissivity_combined"):
            # Extract density from the filename using regular expression
            match = re.search(r'_([\d.e+-]+)_', filename)
            if match:
                file_density = float(match.group(1))
                
                # Calculate the absolute difference between target and file density
                density_difference = abs(target_log_density - file_density)
                
                # Check if this file is a better match than the previous ones
                if density_difference < min_density_difference:
                    min_density_difference = density_difference
                    matching_file = directory+filename

    return matching_file

def interp_emis_temp(original_array):

    # Interpolate into array with size 401
    new_size = 101
    new_indices = np.linspace(0, len(original_array) - 1, new_size)
    interpolated_array = np.interp(new_indices, np.arange(len(original_array)), original_array)
    return interpolated_array

class ashmcmc:
    def __init__(self, filename):
        # self.name = "ashmcmc"
        # self.version = "0.5"
        # self.author = "Andy S.H. To"
        # self.email = "andysh.to@esa.int"
        self.filename = filename
        self.ash = asheis(filename)
        self.outdir = 'results/'+filename.split('/')[-1].replace('.data.h5', '')

    # def fit_data_parallel(self, i):
    #     if i[:2] == 'fe':
    #         print(i)
    #         if self.ash.check_window(i) != None:  # Provide the 'line' argument
    #             print('checked_window')
    #             intensity = self.ash.get_intensity(i, outdir=self.outdir, mcmc=True, plot=False)
    #             return i, intensity

    def fit_data(self, **kwargs):
        from tqdm import tqdm
        from eispac import read_cube
        # Fit data in parallel
        # Returns a map array of intensities
        Lines = []
        dim = read_cube(self.filename).dimensions.value
        dem_num = 1
        for line in list(self.ash.dict.keys()):
            if line[:2] == 'fe':
                if self.ash.check_window(line) != None: 
                    dem_num += 1 
                    Lines.append(line)

        Intensities = np.zeros((int(dim[0]), int(dim[1]), dem_num))    
        Int_error = np.zeros((int(dim[0]), int(dim[1]), dem_num))    
        print(f'------------------------------Found {dem_num} usable lines------------------------------')
        print(f'Found {dem_num} usable lines for DEM')
        for ind, line in tqdm(enumerate(Lines)):
            Intensities[:, :, ind], Int_error[:, :, ind] = self.ash.get_intensity(line, outdir=self.outdir, mcmc=True, **kwargs)

        return Lines, Intensities, Int_error

    def read_density(self,calib=True, **kwargs):
        # Read density from asheis object
        # Returns an array of log densities
        ldens = self.ash.get_density(outdir=self.outdir, mcmc=True, calib=calib, **kwargs)

        return ldens
    
    def read_emissivity(self, ldens, abund_file = 'emissivities_sun_photospheric_2015_scott'):
        from scipy.io import readsav
        import astropy.units as u
        # Find matching file based on density
        emis_file = readsav(find_matching_file(ldens, abund_file=abund_file))
         # print(find_matching_file(ldens, abund_file=abund_file))
        logt = 10**emis_file['logt_interpolated']*u.K
        emis = emis_file['emissivity_combined']
        linenames = emis_file['linenames'].astype(str)

        return logt, emis, linenames

    def emis_filter(self, emis, linenames, obs_Lines):
        import numpy as np
        # Filter emissivity based on specified lines
        emis_sorted = np.zeros((len(obs_Lines),101))
        for ind, line in enumerate(obs_Lines):
            emis_sorted[ind, :] = emis[np.where(linenames == line),:]

        return emis_sorted
    
    
if __name__ == "__main__":
    ash_mcmc = ashmcmc()
    ash_mcmc.fit_data()

