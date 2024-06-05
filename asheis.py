# a lot of information are gathered through Will Barnes Hinode-14 sunpy tutorial - It will be a good manor to cite the sunpy team

from pathlib import Path
import re
import eispac
import sunpy
from matplotlib import colors
import matplotlib.pyplot as plt
from datetime import datetime
# from alpha_code import alpha, alpha_map
import platform
from astropy.visualization import ImageNormalize, quantity_support
from demcmc_FIP.eis_calibration.eis_calib_2014 import calib_2014
from demcmc_FIP.eis_calibration.eis_calib_2023 import calib_2023
import configparser

def load_plotting_routine():
    fig = plt.figure()
    fig.set_dpi(300)
    BIGGER_SIZE=9
    SMALLER_SIZE=8
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALLER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def load_axes_labels():
    plt.xlabel('x (arcsec)')
    plt.ylabel('y (arcsec)')


class asheis:
    '''
    The fitting routine auto align array and map to the 195.119 window 
    '''
    def __init__(self, filename, ncpu='max', rebin=False):
        self.filename = filename
        self.dict = {
            "fe_8_185.21" : ["fe_08_185_213.1c.template.h5",0],
            "fe_8_186.60" : ["fe_08_186_601.1c.template.h5",0],
            "fe_9_188.50" : ["fe_09_188_497.3c.template.h5",0],
            "fe_9_197.86" : ["fe_09_197_862.1c.template.h5",0],
            "fe_10_184.54" : ["fe_10_184_536.1c.template.h5",0],
            "fe_11_188.22" : ["fe_11_188_216.2c.template.h5",0],
            "fe_11_188.30" : ["fe_11_188_299.2c.template.h5",1],
            "fe_12_186.88" : ["fe_12_186_880.1c.template.h5",0],
            "fe_12_195.12" : ["fe_12_195_119.2c.template.h5",0],
            "fe_12_192.39" : ["fe_12_192_394.1c.template.h5",0],
            "fe_13_202.04" : ["fe_13_202_044.1c.template.h5",0],
            "fe_13_203.83" : ["fe_13_203_826.2c.template.h5",2],
            "fe_14_264.79" : ["fe_14_264_787.1c.template.h5",0],
            "fe_14_270.52" : ["fe_14_270_519.2c.template.h5",1],
            "fe_15_284.16" : ["fe_15_284_160.2c.template.h5",1],
            "fe_16_262.98" : ["fe_16_262_984.1c.template.h5",0],
            "fe_17_254.87" : ["fe_17_254_870.2c.template.h5",0],
            "fe_22_253.10" : ["fe_22_253_170.1c.template.h5",0],
            "fe_23_263.76" : ["fe_23_263_760.1c.template.h5",0],
            "fe_24_255.10" : ["fe_24_255_100.2c.template.h5",1],
            "ca_14_193.87" :["ca_14_193_874.6c.template.h5",1],
            "ar_11_188.81" :["ar_11_188_806.3c.template.h5",2],
            "ar_14_194.40" : ["ar_14_194_396.6c.template.h5",5],
            "si_10_258.37" :["si_10_258_375.1c.template.h5",0],
            "s_10_264.23" : ["s__10_264_233.1c.template.h5",0],
            "s_11_188.68" : ["s__11_188_675.3c.template.h5",1],
            "s_13_256.69" : ["s__13_256_686.1c.template.h5",0],
        }
        self.ncpu = ncpu
        self.rebin = rebin

        config_obj = configparser.ConfigParser()
        config_obj.read("demcmc_FIP/configfile.ini")
        directories = config_obj["directories"]
        main_dir = directories['main_dir']
        self.dens_dir = main_dir+'density'

#        if platform.system() == "Linux":
##            self.dens_dir = '/home/staff/daithil/work/python_output/FIP/density'
#            self.dens_dir = '/disk/solar2/dml/python/FIP/density'
#        elif platform.system() == "Darwin":
#            self.dens_dir = '/Users/dml/python_output/FIP/density'


    def check_window(self, line):
        print(f'Checking {line}')
        template_name=self.dict[f'{line}'][0]
        template = eispac.read_template(eispac.data.get_fit_template_filepath(template_name))
        cube = eispac.read_cube(self.filename, window=template.central_wave)
        return cube

    def fit_data(self,line,product,refit, outdir):
        from eispac.instr import ccd_offset
        template_name=self.dict[f'{line}'][0]
        # print(self.filename.replace("data.h5",template_name))
        if template_name != 'fe_13_203_826.2c.template.h5':
            template = eispac.read_template(eispac.data.get_fit_template_filepath(template_name))
        else:
            template = eispac.read_template('demcmc_FIP/eis_density/fe_13_203_830.3c.template.h5')
            template_name = 'fe_13_203_830.3c.template.h5'

        path = Path(f'{self.filename}'.replace("data.h5",template_name).replace(".template",f"-{self.dict[f'{line}'][1]}.fit"))
        # if self rebin != False:
        print(path)
        if path.is_file() == False or refit==True or self.rebin!=False:
            cube = eispac.read_cube(self.filename, window=template.central_wave)
            if self.rebin != False:
                print('Rebinning')
                cube = cube.smooth_cube(self.rebin)
            fit_res = eispac.fit_spectra(cube, template, ncpu=self.ncpu)
            fit_res.fit[f'{product}'] = fit_res.shift2wave(fit_res.fit[f'{product}'],wave=195.119)
            disp = ccd_offset(195.119) - ccd_offset(fit_res.fit['wave_range'].mean())
            fit_res.meta['mod_index']['crval2'] = float(fit_res.meta['mod_index']['crval2'] - disp)
            save_filepaths = eispac.save_fit(fit_res)
        else:
            fit_res=eispac.read_fit(path)

        return fit_res
    
    
    def directory_setup(self, amap, line, outdir):
        # Set up directory and save fit
        date = amap.date.strftime("%Y_%m_%d__%H_%M_%S")
        Path(f'{outdir}/images/fits/').mkdir(parents=True, exist_ok=True)
        Path(f'{outdir}/images/{amap.measurement.lower().split()[-1]}/{line}/').mkdir(parents=True, exist_ok=True)
        # print(date)
        # print(amap.measurement.lower().split())
        # print(f'{date}')
        amap.save(f"{outdir}/images/{amap.measurement.lower().split()[-1]}/{line}/eis_{date}_{'_'.join(amap.measurement.lower().split())}.fits", overwrite=True)
        return date
    
    def plot_int_map(self, date, amap, line, outdir, colorbar=False, savefig=True, **kwargs):
        load_plotting_routine()
        amap.plot(**kwargs)
        if colorbar==True: plt.colorbar() 
        load_axes_labels()
        # plt.savefig(f'{date}/eis_{m.measurement.lower().replace(" ","_").replace(".","_")}.png')
        if savefig==True: plt.savefig(f'{outdir}/images/{amap.measurement.lower().split()[-1]}/{line}/eis_{date}_{amap.measurement.lower().replace(" ","_").replace(".","_")}.png')
        # plt.savefig(f'images/{amap.measurement.lower().split()[-1]}/eis_{date}_{amap.measurement.lower().replace(" ","_").replace(".","_")}.png')

    def plot_fip_map(self, date, amap, outdir, colorbar=True, savefig=True):
        load_plotting_routine()

        norm = colors.Normalize(vmin=0, vmax=4)
        amap.plot_settings['norm'] = norm
        amap.plot_settings['cmap'] = 'RdYlBu'
        amap.plot()

        if colorbar==True: plt.colorbar() 
        load_axes_labels()
        if savefig==True: plt.savefig(f'{outdir}/images/{date}_{amap.measurement.lower().replace(" ","_").replace(".","_")}.png')

    def get_intensity(self, line, outdir='', refit=False, plot=True, mcmc=False, calib=True):
        fit_res = self.fit_data(line,'int',refit, outdir) # Get fitdata
        m = fit_res.get_map(self.dict[f'{line}'][1],measurement='intensity') # From fitdata get map
        if calib: # Calibrate data using NRL calibration (Warren et al. 2014)
            m, calib_ratio = calib_2014(m, ratio=True)
            print(f'---------------------Calibrated using Warren et al. 2014; Ratio: {calib_ratio}---------------------')
        date = self.directory_setup(m,line,outdir) # Creating directories
        if plot == True: self.plot_int_map(date, m, line, outdir) # Plot maps
        if mcmc:
            if calib:
                m_error = fit_res.fit['err_int'][:,:,self.dict[f'{line}'][1]]*calib_ratio
            else:  
                m_error = fit_res.fit['err_int'][:,:,self.dict[f'{line}'][1]]
            return m.data, m_error
        else:
            return m
        
    def get_velocity(self, line, outdir='',vmin=-10,vmax=10, refit=False, plot=True):
        fit_res = self.fit_data(line,'vel',refit, outdir)
        m = fit_res.get_map(component = self.dict[f'{line}'][1],measurement='velocity')
        date = self.directory_setup(m,line,outdir)
        m.plot_settings['norm'] = ImageNormalize(vmin=vmin,vmax=vmax) # adjusting the velocity saturation
        if plot == True: self.plot_int_map(date, m, line, colorbar=True)
        return m
        
    def get_width(self, line, outdir='', refit=False, plot=True):
        fit_res = self.fit_data(line,'vel', outdir)
        m = fit_res.get_map(component = self.dict[f'{line}'][1],measurement='width')
        date = self.directory_setup(m,line)
        if plot == True: self.plot_int_map(date, m, line, colorbar=True)
        return m
    
    def get_density(self, outdir='', refit=False, plot=True, mcmc=False, **kwargs):
        from scipy.io import readsav
        import numpy as np
        from astropy.visualization import ImageNormalize
        import astropy.units as u

        density_ratios = readsav(f'{self.dens_dir}/density_ratios_fe_13_203_82_202_04_from_IDL.sav')['smooth_rat']
        density_values = readsav(f'{self.dens_dir}/density_ratios_fe_13_203_82_202_04_from_IDL.sav')['smooth_den']

        m_nom = self.get_intensity('fe_13_203.83', outdir, plot=False, **kwargs)
        m_denom = self.get_intensity('fe_13_202.04', outdir, plot=False, **kwargs)
        obs_ratio = m_nom.data / m_denom.data

        for i in range(obs_ratio.shape[0]):
            for j in range(obs_ratio.shape[1]):
                obs = obs_ratio[i, j]
                closest_index = np.argmin(np.abs(density_ratios - obs))
                obs_ratio[i, j] = density_values[closest_index]

        m = sunpy.map.Map(obs_ratio, m_nom.meta)
        m.meta['measrmnt'] = 'density'
        m.meta['bunit'] = '' # probably want to change this to 1/cm3 in the future
        # m.meta['line_id'] = 'Fe XIII'
        m.plot_settings['norm'] = ImageNormalize(vmin=8, vmax=10)
        if mcmc:
            return m.data
        else:
            return m



    def get_composition(self, linepair, outdir='', vmin=0, vmax=4, **kwargs):
        '''
        This quick look composition code is incomplete and probably doesn't work. Especially be careful of shift2wave code.
        '''
        line_databases = {
            "SiS": ['si_10_258', 's_10_264', 'Si X-S X'],
            "CaAr": ['ca_14_193', 'ar_14_194', 'Ca XIV-Ar XIV'],
            "FeS": ['fe_16_262', 's_13_256', 'Fe XVI-S XIII'],
            "SAr": ['s_13_256', 'ar_14_194', 'S XIII-Ar XIV'],
            "SAr2": ['s_11_188', 'ar_11_188', 'S XI-Ar XI'],
            # "FeAr": ['fe_16_262', 'ar_11_188', 'Fe XVI-Ar XI'], wrong dignostic
            # Add more line pairs as needed
        }

        if linepair not in line_databases:
            print('No line database can be found. Add your line in code.')
            return

        lines = line_databases[linepair]

        m_nom = self.get_intensity(lines[0], outdir)
        m_denom = self.get_intensity(lines[1], outdir)
        m_ref = self.get_intensity('fe_12_195', outdir, plot=False)

        m_fip = sunpy.map.Map(m_nom.data / m_denom.data, m_ref.meta)
        m_fip.meta['line_id'] = lines[2]
        m_fip.meta['measrmnt'] = 'composition'
        m_fip.meta.pop('bunit', None)

        date = self.directory_setup(m_fip, lines[2])
        
        # m_fip.save(f"images/{m_fip.measurement.lower().split()[-1]}/{lines[2]}/eis_{date}_{'_'.join(m_fip.measurement.lower().split())}.fits", overwrite=True)
        self.plot_fip_map(date, m_fip, outdir, vmin=vmin, vmax=vmax, norm=colors.Normalize(), cmap='CMRmap')
        
if __name__ == '__main__':
    load_plotting_routine()
    load_axes_labels()
    # def get_alpha(self, line,vmin=0,vmax=0.3,cmap='viridis'):
    #     template_name=self.dict[f'{line}'][0]
    #     template = eispac.read_template(eispac.data.get_fit_template_filepath(template_name))
    #     fit_res = self.fit_data(line,'int')
    #     data_cube = eispac.read_cube(self.filename, window=template.central_wave)
    #     m = fit_res.get_map(self.dict[f'{line}'][1],measurement='intensity')
    #     m.meta['measrmnt'] = 'Alpha'
    #     # mapvalue = alpha_map(fit_res, data_cube, self.ncpu)
    #     m = sunpy.map.Map(alpha_map(fit_res, data_cube, self.ncpu), m.meta)
    #     m.plot_settings['norm'] = ImageNormalize(vmin=vmin,vmax=vmax)
    #     m.plot_settings['cmap']=plt.get_cmap(cmap)
    #     date = self.directory_setup(m)
    #     self.plot_map(date, m, colorbar=True)

