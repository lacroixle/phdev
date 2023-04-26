#!/usr/bin/env python3

import pathlib
from shutil import copyfile
import pickle
import tarfile
import json
from shutil import rmtree
import os

from ztfquery.io import get_file
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np
from ztfimg.utils.tools import ccdid_qid_to_rcid

from utils import ListTable, quadrant_name_explode, quadrant_width_px, quadrant_height_px, j2000mjd


class Exposure:
    def __init__(self, lightcurve, name, path=None):
        self.__lightcurve = lightcurve
        self.__name = name
        self.__year, self.__month, self.__day, self.__field, self.__filterid, self.__ccdid, self.__qid = quadrant_name_explode(name)
        if path is None:
            self.__path = self.__lightcurve.path.joinpath(name)
        else:
            self.__path = path

    @property
    def lightcurve(self):
        return self.__lightcurve

    @property
    def name(self):
        return self.__name

    @property
    def year(self):
        return self.__year

    @property
    def month(self):
        return self.__month

    @property
    def day(self):
        return self.__day

    @property
    def field(self):
        return self.__field

    @property
    def filterid(self):
        return self.__filterid

    @property
    def ccdid(self):
        return self.__ccdid

    @property
    def rcid(self):
        return ccdid_qid_to_rcid(self.__ccdid, self.__qid)

    @property
    def qid(self):
        return self.__qid

    @property
    def path(self):
        return self.__path

    @property
    def mjd(self):
        return float(self.exposure_header['obsmjd'])

    @property
    def wcs(self):
        return WCS(self.exposure_header)

    def retrieve_exposure(self):
        image_path = pathlib.Path(get_file(self.name + "_sciimg.fits", downloadit=False))
        if not image_path.exists():
            raise FileNotFoundError("Science image at {} not found on disk!".format(image_path))

        copyfile(image_path, self.path.joinpath("calibrated.fits"))
        return image_path

    def update_exposure_header(self):
        if self.path.joinpath("calibrated.fits").exists():
            with fits.open(self.path.joinpath("calibrated.fits")) as hdul:
                hdul[0].header.tofile(self.path.joinpath("calibrated.header"), overwrite=True)

    @property
    def exposure_header(self):
        if self.path.joinpath("calibrated.fits").exists():
           with fits.open(self.path.joinpath("calibrated.fits")) as hdul:
               return hdul[0].header
        elif self.path.joinpath("calibrated.header").exists():
            with fits.open(self.path.joinpath("calibrated.header")) as hdul:
                return hdul[0].header
        else:
            raise FileNotFoundError("Could not find calibrated.fits or calibrated.header for exposure {}!".format(self.name))

    def get_catalog(self, cat_name, key=None):
        if not isinstance(cat_name, pathlib.Path):
            cat_name = pathlib.Path(cat_name)

        if cat_name.suffix == ".list" or cat_name.suffix == ".dat":
            cat = ListTable.from_filename(self.path.joinpath(cat_name.name))
        elif cat_name.suffix == ".parquet":
            cat = pd.read_parquet(self.path.joinpath(cat_name.name))
        elif cat_name.suffix == ".hd5":
            cat = pd.read_hdf(self.path.joinpath(cat_name.name), key=key)
        else:
            raise ValueError("Catalog {} extension {} not recognized!".format(cat_name.name, cat_name.suffix))

        return cat

    def get_ext_catalog(self, cat_name, pm_correction=True, project=False):
        if cat_name not in self.lightcurve.ext_star_catalogs_name:
            raise ValueError("External catalogs are {}, not {}")

        ext_cat_df = self.lightcurve.get_ext_catalog(cat_name)

        if pm_correction:
            obsmjd = self.mjd
            if cat_name == 'gaia':
                ext_cat_df['ra'] = ext_cat_df['ra']+(obsmjd-j2000mjd)*ext_cat_df['pmRA']
                ext_cat_df['dec'] = ext_cat_df['dec']+(obsmjd-j2000mjd)*ext_cat_df['pmDE']
            elif cat_name == 'ps1':
                pass
            else:
                raise NotImplementedError()

        if project:
            skycoords = SkyCoord(ra=ext_cat_df['ra'].to_numpy(), dec=ext_cat_df['dec'].to_numpy(), unit='deg')
            stars_x, stars_y = skycoords.to_pixel(self.wcs)
            ext_cat_df['x'] = stars_x
            ext_cat_df['y'] = stars_y

        return ext_cat_df

    def get_matched_catalog(self, cat_name):
        if cat_name != 'aperstars' and cat_name != 'psfstars':
            raise ValueError("Only matched catalog are \'aperstars\' and \'psfstars\', not {}".format(cat_name))

        with pd.HDFStore(self.path.joinpath("cat_indices.hd5"), 'r') as hdfstore:
            cat_indices = hdfstore.get('{}_indices'.format(cat_name))
            ext_cat_indices = hdfstore.get('cat_indices')

        if cat_name == 'aperstars':
            cat_name = 'standalone_stars'

        cat_df = self.get_catalog("{}.list".format(cat_name)).df
        return cat_df.iloc[cat_indices].reset_index(drop=True).iloc[ext_cat_indices].reset_index(drop=True)

    def get_matched_ext_catalog(self, cat_name, pm_correction=True, project=False):
        if cat_name not in self.lightcurve.ext_star_catalogs_name:
            raise ValueError("External catalogs are {}, not {}".format(self.lightcurve.ext_star_catalogs_name, cat_name))

        ext_cat_df = self.lightcurve.get_ext_catalog(cat_name)
        with pd.HDFStore(self.path.joinpath("cat_indices.hd5"), 'r') as hdfstore:
            cat_indices = hdfstore.get('ext_cat_indices')['indices']

        ext_cat_df = ext_cat_df.iloc[cat_indices].reset_index(drop=True)

        if pm_correction:
            obsmjd = self.mjd
            if cat_name == 'gaia':
                ext_cat_df['ra'] = ext_cat_df['ra']+(obsmjd-j2000mjd)*ext_cat_df['pmRA']
                ext_cat_df['dec'] = ext_cat_df['dec']+(obsmjd-j2000mjd)*ext_cat_df['pmDE']
            elif cat_name == 'ps1':
                pass
            else:
                raise NotImplementedError()

        if project:
            skycoords = SkyCoord(ra=ext_cat_df['ra'].to_numpy(), dec=ext_cat_df['dec'].to_numpy(), unit='deg')
            stars_x, stars_y = skycoords.to_pixel(self.wcs)
            ext_cat_df['x'] = stars_x
            ext_cat_df['y'] = stars_y

        return ext_cat_df

    def func_status(self, func_name):
        return self.path.joinpath("{}.success".format(func_name)).exists()

    def func_timing(self, func_name):
        timings_path = self.path.joinpath("timings_{}".format(func_name))
        if timings_path.exists():
            with open(timings_path, 'r') as f:
                return json.load(f)

    def get_centroid(self):
        return self.wcs.pixel_to_world_values(np.array([[quadrant_width_px/2., quadrant_height_px/2.]]))[0]


class BaseLightcurve:
    def __init__(self, name, filterid, wd):
        pass


class Lightcurve:
    def __init__(self, name, filterid, wd, exposure_regexp="ztf_*"):
        self.__name = name
        self.__filterid = filterid
        self.__path = wd.joinpath("{}/{}".format(name, filterid))
        self.__noprocess_path = self.__path.joinpath("noprocess")
        self.__driver_path = self.__path.joinpath("smphot_driver")
        self.__ext_catalogs_path = self.__path.joinpath("catalogs")
        self.__astrometry_path = self.__path.joinpath("astrometry")
        self.__photometry_path = self.__path.joinpath("photometry")
        self.__mappings_path = self.__path.joinpath("mappings")
        self.__smphot_path = self.__path.joinpath("smphot")
        self.__smphot_stars_path = self.__path.joinpath("smphot_stars")

        self.__exposures = dict([(exposure_path.name, Exposure(self, exposure_path.name)) for exposure_path in list(self.__path.glob(exposure_regexp))])

    @property
    def path(self):
        return self.__path

    @property
    def name(self):
        return self.__name

    @property
    def filterid(self):
        return self.__filterid

    @property
    def ext_catalogs_path(self):
        return self.__ext_catalogs_path

    @property
    def astrometry_path(self):
        return self.__astrometry_path

    @property
    def photometry_path(self):
        return self.__photometry_path

    @property
    def mappings_path(self):
        return self.__mappings_path

    @property
    def driver_path(self):
        return self.__driver_path

    @property
    def smphot_path(self):
        return self.__smphot_path

    @property
    def smphot_stars_path(self):
        return self.__smphot_stars_path

    @property
    def noprocess_path(self):
        return self.__noprocess_path

    @property
    def exposures(self):
        return self.__exposures

    @property
    def star_catalogs_name(self):
        return ['aperstars', 'psfstars']

    @property
    def ext_star_catalogs_name(self):
        return ['gaia', 'ps1', 'ubercal']

    def get_exposures(self, files_to_check=None, ignore_noprocess=False):
        if files_to_check is None and ignore_noprocess:
            return self.__exposures.values()

        if files_to_check is None:
            files_to_check_list = []
        elif isinstance(files_to_check, str):
            files_to_check_list = [files_to_check]
        else:
            files_to_check_list = files_to_check

        def _check_files(exposure):
            check_ok = True
            for check_file in files_to_check_list:
                if not self.__path.joinpath("{}/{}".format(exposure.name, check_file)).exists():
                    check_ok = False
                    break

            return check_ok

        return list(filter(lambda x: (x.name not in self.get_noprocess() or ignore_noprocess) and _check_files(x), list(self.__exposures.values())))

    def add_noprocess(self, new_noprocess_quadrants):
        noprocess_quadrants = self.get_noprocess()
        noprocess_written = 0
        if isinstance(new_noprocess_quadrants, str) or isinstance(new_noprocess_quadrants, pathlib.Path):
            new_noprocess_quadrants_list = [new_noprocess_quadrants]
        else:
            new_noprocess_quadrants_list = new_noprocess_quadrants

        with open(self.noprocess_path, 'a') as f:
            for new_quadrant in new_noprocess_quadrants_list:
                if new_quadrant not in noprocess_quadrants:
                    f.write("{}\n".format(new_quadrant))
                    noprocess_written += 1

        return noprocess_written

    def get_noprocess(self):
        noprocess = []
        if self.noprocess_path.exists():
            with open(self.noprocess_path, 'r') as f:
                for line in f.readlines():
                    quadrant = line.strip()
                    if quadrant[0] == "#":
                        continue
                    elif self.path.joinpath(quadrant).exists():
                        noprocess.append(quadrant)

        return noprocess

    def reset_noprocess(self):
        if self.noprocess_path().exists():
            self.noprocess_path().unlink()

    def get_ext_catalog(self, cat_name, matched=True):
        if matched:
            catalog_df = pd.read_parquet(self.ext_catalogs_path.joinpath("{}.parquet".format(cat_name)))
        else:
            catalog_df = pd.read_parquet(self.ext_catalogs_path.joinpath("{}_full.parquet".format(cat_name)))

        return catalog_df

    def get_catalogs(self, cat_name, files_to_check=None):
        if files_to_check is None:
            files_to_check = cat_name
        elif isinstance(files_to_check, str) or isinstance(files_to_check, pathlib.Path):
            files_to_check = [cat_name, files_to_check]
        else:
            files_to_check.append(cat_name)

        exposures = self.get_exposures(files_to_check=files_to_check)
        return dict([(exposure.name, exposure.get_catalog(cat_name)) for exposure in exposures])

    def exposure_headers(self):
        exposures = list(filter(lambda x: True, self.get_exposures()))
        return dict([(exposure.name, exposure.exposure_header()) for exposure in exposures])

    def get_reference_exposure(self):
        if not self.__path.joinpath("reference_exposure").exists():
            raise FileNotFoundError("{}-{}: reference exposure has not been determined!".format(self.name, self.filterid))

        with open(self.__path.joinpath("reference_exposure"), 'r') as f:
            return f.readline().strip()

    def extract_exposure_catalog(self):
        exposures = []
        for exposure in self.get_exposures():
            header = exposure.exposure_header
            exposure_dict = {}
            exposure_dict['name'] = exposure.name
            exposure_dict['airmass'] = header['airmass']
            exposure_dict['mjd'] = header['obsmjd']
            exposure_dict['seeing'] = header['seeing']
            exposure_dict['ha'] = header['hourangd'] #*15
            exposure_dict['ha_15'] = 15.*header['hourangd']
            exposure_dict['lst'] = header['oblst']
            exposure_dict['azimuth'] = header['azimuth']
            exposure_dict['dome_azimuth'] = header['dome_az']
            exposure_dict['elevation'] = header['elvation']
            exposure_dict['z'] = 90. - header['elvation']
            exposure_dict['telra'] = header['telrad']
            exposure_dict['teldec'] = header['teldecd']

            exposure_dict['field'] = header['dbfield']
            exposure_dict['ccdid'] = header['ccd_id']
            exposure_dict['qid'] = header['amp_id']
            exposure_dict['rcid'] = header['dbrcid']

            exposure_dict['fid'] = header['dbfid']

            exposure_dict['temperature'] = header['tempture']
            exposure_dict['head_temperature'] = header['headtemp']
            exposure_dict['ccdtemp'] = float(header['ccdtmp{}'.format(str(header['ccd_id']).zfill(2))])

            exposure_dict['wind_speed'] = header['windspd']
            exposure_dict['wind_dir'] = header['winddir']
            exposure_dict['dewpoint'] = header['dewpoint']
            exposure_dict['humidity'] = header['humidity']
            exposure_dict['wetness'] = header['wetness']
            exposure_dict['pressure'] = header['pressure']

            exposure_dict['crpix1'] = header['crpix1']
            exposure_dict['crpix2'] = header['crpix2']
            exposure_dict['crval1'] = header['crval1']
            exposure_dict['crval2'] = header['crval2']
            exposure_dict['cd_11'] = header['cd1_1']
            exposure_dict['cd_12'] = header['cd1_2']
            exposure_dict['cd_21'] = header['cd2_1']
            exposure_dict['cd_22'] = header['cd2_2']

            exposure_dict['skylev'] = header['sexsky']
            exposure_dict['sigma_skylev'] = header['sexsigma']

            exposures.append(exposure_dict)

        return pd.DataFrame(exposures).set_index('name')

    def extract_star_catalog(self, catalog_names, project=False):
        for catalog_name in catalog_names:
            if catalog_name not in (self.star_catalogs_name+self.ext_star_catalogs_name):
                raise ValueError("Star catalog name \'{}\' does not exists!".format(catalog_name))

        catalog_list = []
        files_to_check = ["indices.hd5"]
        exposures = self.get_exposures(files_to_check=["cat_indices.hd5"])
        name_set = False

        for catalog_name in catalog_names:
            catalogs = []

            for exposure in exposures:
                if catalog_name in self.star_catalogs_name:
                    catalog_df = exposure.get_matched_catalog(catalog_name)
                else:
                    catalog_df = exposure.get_matched_ext_catalog(catalog_name, project=project)

                if not name_set:
                    catalog_df.insert(0, 'exposure', exposure.name)

                catalogs.append(catalog_df)

            catalogs_df = pd.concat(catalogs)

            if len(catalog_names) > 1:
                if not name_set:
                    catalogs_df.columns = ['exposure'] + ['{}_{}'.format(catalog_name, column_name) for column_name in catalogs_df.columns[1:]]
                else:
                    catalogs_df.columns = ['{}_{}'.format(catalog_name, column_name) for column_name in catalogs_df.columns]

            if not name_set:
                name_set = True

            catalog_list.append(catalogs_df)

        return pd.concat(catalog_list, axis='columns').reset_index().rename(columns={'index': 'cat_index'})

    def func_status(self, func_name):
        if self.path.joinpath("{}.success".format(func_name)).exists():
            return 1.
        elif self.path.joinpath("{}.fail".format(func_name)).exists():
            return 0.
        else:
            exposures = self.get_exposures()
            if len(exposures) == 0:
                return 0.

            success = [exposure.func_status(func_name) for exposure in exposures]
            return sum(success)/len(exposures)

    def func_timing(self, func_name):
        map_timing = {'start': float('inf'), 'end': -float('inf'), 'elapsed': 0.}
        reduce_timing = {'start': float('inf'), 'end': -float('inf'), 'elapsed': 0.}

        timing_path = self.path.joinpath("timings_{}".format(func_name))
        if timing_path.exists():
            with open(timing_path, 'r') as f:
                reduce_timing = json.load(f)

        map_timings = [exposure.func_timing(func_name) for exposure in self.get_exposures()]
        map_timings = list(filter(lambda x: x is not None, map_timings))

        if len(map_timings) > 0:
            map_timings_df = pd.DataFrame(map_timings)
            map_timing = {'start': map_timings_df['start'].min(),
                        'end': map_timings_df['end'].max()}
            map_timing['elapsed'] = map_timing['end'] - map_timing['start']

        total_timing = {'start': min([map_timing['start'], reduce_timing['start']]),
                        'end': max([map_timing['end'], map_timing['start']])}
        total_timing['elapsed'] = total_timing['end'] - total_timing['start']

        return {'map': map_timing,
                'reduce': reduce_timing,
                'total': total_timing}

    def compress(self):
        """
        Compress lightcurve into a big hdf and tar files (more efficient storage, limiting small files)
        If funcs is provided, also fill success rates and timings for each pipeline step.
        funcs is expected to be a list of strings, each being a pipeline step.
        """

        exposures = self.get_exposures(ignore_noprocess=True)

        # Store catalogs into one big HDF file
        catalogs_to_store = ['standalone_stars', 'psfstars']
        with pd.HDFStore(self.path.joinpath("catalogs.hdf"), 'w') as hdfstore:
            # Store exposure catalogs
            for exposure in exposures:
                for catalog in catalogs_to_store:
                    try:
                        cat = exposure.get_catalog(catalog+".list")
                    except FileNotFoundError:
                        continue

                    for key in cat.header.keys():
                        hdfstore.put('{}/{}/header/{}'.format(exposure.name, catalog, key), pd.Series(data=cat.header[key]))

                    hdfstore.put('{}/{}/df'.format(exposure.name, catalog), cat.df)

        # Store headers
        headers = dict([(exposure.name, exposure.exposure_header) for exposure in exposures])
        with open(self.path.joinpath("headers.pickle"), 'wb') as f:
            pickle.dump(headers, f)

        # Archive all files/folders
        tar_path = self.path.joinpath("lightcurve.tar")
        tar_path.unlink(missing_ok=True)

        files = list(self.path.glob("*"))
        files.remove(self.path.joinpath("catalogs.hdf"))
        files.remove(self.path.joinpath("headers.pickle"))

        tar = tarfile.open(self.path.joinpath("lightcurve.tar"), 'w')
        for f in files:
            tar.add(f, f.relative_to(self.path))
        tar.close()

        # Delete all files
        for f in files:
            if f.is_dir():
                rmtree(f)
            else:
                f.unlink()

    def uncompress(self, keep_compressed_files=True):
        if not self.path.joinpath("lightcurve.tar").exists():
            return

        tar = tarfile.open(self.path.joinpath("lightcurve.tar"), 'r')
        tar.extractall(path=self.path)
        tar.close()

        if not keep_compressed_files:
            self.path.joinpath("catalogs.hdf").unlink(missing_ok=True)
            self.path.joinpath("headers.pickle").unlink(missing_ok=True)
            self.path.joinpath("lightcurve.tar").unlink(missing_ok=True)


class CompressedLightcurve:
    def __init__(self, name, filterid, wd):
        pass
