import os.path as opth
import glob
import numpy as np
import netCDF4 as ncf

from itertools import izip


def compile_posterior_for_lim(dir, varnames):
    if opth.isdir(dir):
        iter_fldrs = glob.glob(opth.join(dir, 'r*'))
    else:
        raise IOError('Directory does not exist: {}'.format(dir))

    ens_filename = 'ensemble_mean_{}.npz'
    tmp_fldr = iter_fldrs[0]
    useable_vars = []
    lat_out = np.array([])
    lon_out = np.array([])
    sptl_len = 0
    for varname in varnames:
        ens_file = opth.join(tmp_fldr, ens_filename.format(varname))
        if opth.exists(ens_file):
            useable_vars.append(varname)

            tmp = np.load(ens_file)
            time = tmp['years']

            if 'lat' in tmp.keys() and 'lon' in tmp.keys():
                lat = tmp['lat']
                lon = tmp['lon']
                sptl_len += lat.size
            else:
                lat = np.array([0])  # weighted as 1 in area weighting
                lon = np.array([0])
                sptl_len += 1

            lat_out = np.concatenate((lat_out, lat.flatten()))
            lon_out = np.concatenate((lon_out, lon.flatten()))

    ens_out = np.zeros((len(iter_fldrs), len(time), sptl_len))

    for k, (fldr, ens_itr_out) in enumerate(izip(iter_fldrs, ens_out)):

        print 'Processing iteration {:d}/{:d}'.format(k+1, len(iter_fldrs))

        ens_file = opth.join(fldr, ens_filename)
        var_start = 0
        for varname in useable_vars:

            curr_file = ens_file.format(varname)
            npdict = np.load(curr_file)
            dat = npdict['xam']

            if dat.ndim > 1:
                lat = npdict['lat']
                dat = dat.reshape(len(time), lat.size)
            else:
                dat = dat[:, None]

            var_end = var_start + dat.shape[-1]
            ens_itr_out[:, var_start:var_end] = dat

            var_start = var_end

    # Calculate mean over iterations
    itr_mean = ens_out.mean(axis=0)

    return itr_mean, time, lat_out, lon_out


def save_to_netcdf(data, time, lat, lon, outfilename, outvarname):

    try:
        f = ncf.Dataset(outfilename, 'w')
        f.createDimension('time', len(time))
        f.createDimension('lat', len(lat))
        f.createDimension('lon', len(lon))
        ftime = f.createVariable('time', 'i4', ('time',))
        flat = f.createVariable('lat', 'f4', ('lat',))
        flon = f.createVariable('lon', 'f4', ('lon',))
        fdata = f.createVariable(outvarname, 'f8', ('time', 'lat', 'lon'))

        ftime[:] = time
        flat[:] = lat
        flon[:] = lon
        fdata[:] = data
    finally:
        try:
            f.close()
        except NameError:
            pass


def main():
    workdir = '/home/disk/kalman2/wperkins/LMR_output/archive'
    nexp = 'production_gis_mpi_pagesall_0.75'

    use_vars = ['tas_sfc_Amon', 'zg_500hPa_Amon', 'AMOCindex_Omon']

    expdir = opth.join(workdir, nexp)
    [compiled_recon,
     years,
     lat,
     lon] = compile_posterior_for_lim(expdir, use_vars)

    outfile = opth.join(expdir, 'lim_recon_compiled.nc')
    save_to_netcdf(compiled_recon, years, lat, lon, outfile, 'state')


if __name__ == '__main__':
    main()
