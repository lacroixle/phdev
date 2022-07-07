#!/usr/bin/env python3

import pathlib
from collections.abc import Iterable

import numpy as np
from croaks.match import NearestNeighAssoc
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u


filtercodes = ['zg', 'zr', 'zi']
quadrant_width_px, quadrant_height_px = 3072, 3080

ztf_longitude = -116.8598 # deg
ztf_latitude = 33.3573 # deg E
ztf_altitude = 1668. # m

gaiarefmjd = Time(2015.5, format='byear').mjd


def cat_to_ds9regions(cat, filename, radius=5., color='green'):
    with open(filename, 'w') as f:
        f.write("global color={} dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n".format(color))
        f.write("image\n")
        for c in cat:
            f.write("circle {} {} 5\n".format(c['x'], c['y']))


def match_pixel_space(refcat, cat, radius=1.):
    def _euclidian(x1, y1, x2, y2):
        return np.sqrt((np.array(x1) - np.array(x2)) ** 2 + (np.array(y1) - np.array(y2)) ** 2)

    assoc = NearestNeighAssoc(first=[refcat['x'], refcat['y']], radius=radius)
    index = assoc.match(cat['x'], cat['y'], metric=_euclidian)
    return index


def get_cat_size(catalog_filename):
    with open(catalog_filename, 'r') as f:
        _, cat = read_list(f)

    return len(cat)


def apply_space_motion(ra, dec, pm_ra, pm_dec, refmjd, newmjd):
    reftime = Time(refmjd, format='mjd')
    newtime = Time(newmjd, format='mjd')
    #coords = SkyCoord(ra*u.deg, dec*u.deg, pm_ra_cosdec=pm_ra*np.cos(dec/180.*np.pi)*u.mas/u.year, pm_dec=pm_dec*u.mas/u.year, obstime=reftime)
    coords = SkyCoord(ra*u.deg, dec*u.deg, pm_ra_cosdec=np.zeros_like(ra)*u.mas/u.year, pm_dec=np.zeros_like(ra)*u.mas/u.year, obstime=reftime)
    coords.apply_space_motion(newtime)
    return np.stack([coords.frame.data.lon.value, coords.frame.data.lat.value], axis=1)

def get_wcs_from_quadrant(quadrant_path):
    with fits.open(quadrant_path.joinpath("calibrated.fits")) as hdul:
        wcs = WCS(hdul[0].header)

    return wcs


def get_mjd_from_quadrant_path(quadrant_path):
    with fits.open(quadrant_path.joinpath("calibrated.fits")) as hdul:
        return hdul[0].header['obsmjd']


def get_header_from_quadrant_path(quadrant_path, key=None):
    with fits.open(quadrant_path.joinpath("calibrated.fits")) as hdul:
        if key is None:
            return hdul[0].header
        else:
            return hdul[0].header[key]


def sc_ra(skycoord):
    return skycoord.frame.data.lon.value


def sc_dec(skycoord):
    return skycoord.frame.data.lat.value


def contained_in_exposure(objects, wcs, return_mask=False):
    width, height = wcs.pixel_shape

    if isinstance(objects, pd.DataFrame):
        mask = (objects['x'] >= 0.) & (objects['x'] < width) & (objects['y'] >= 0.) & (objects['y'] < height)
    else:
        top_left = [0., height]
        top_right = [width, height]
        bottom_left = [0., 0.]
        bottom_right = [width, 0]

        tl_radec = wcs.pixel_to_world(*top_left)
        tr_radec = wcs.pixel_to_world(*top_right)
        bl_radec = wcs.pixel_to_world(*bottom_left)
        br_radec = wcs.pixel_to_world(*bottom_right)

        tl = (max([sc_ra(tl_radec), sc_ra(bl_radec)]), min([sc_dec(tl_radec), sc_dec(tr_radec)]))
        br = (min([sc_ra(tr_radec), sc_ra(br_radec)]), max([sc_dec(bl_radec), sc_dec(br_radec)]))

        mask = (sc_ra(objects) < tl[0]) & (sc_ra(objects) > br[0]) & (sc_dec(objects) > tl[1]) & (sc_dec(objects) < br[1])

    if return_mask:
        return mask
    else:
        return objects[mask]


def get_ref_quadrant_from_driver(driver_path):
    if not driver_path.exists():
        return

    with open(driver_path, 'r') as f:
        found_phoref = False
        line = f.readline()
        while line:
            if "PHOREF" in line[:-1]:
                found_phoref = True
                break

            line = f.readline()

        if not found_phoref:
            return

        reference_quadrant = f.readline()

        if reference_quadrant:
            return pathlib.Path(reference_quadrant[:-1]).name


def poly2d_from_file(filename):
    with open(filename, 'r') as f:
        f.readline()
        deg_str = f.readline()[:-1]
        degree = int(deg_str.split(" ")[1])
        coeff_str = " ".join(f.readline()[:-1].split())
        coeffs = list(map(float, coeff_str.split(" ")))

    coeffs_1 = coeffs[:int(len(coeffs)/2)]
    coeffs_2 = coeffs[int(len(coeffs)/2):]

    def _extract_coeffs(coeffs):
        idx = 0
        c = np.zeros([degree, degree])
        for d in range(degree):
            p, q = d, 0
            while p >= 0:
                c[p, q] = coeffs[idx]
                idx += 1
                p -= 1
                q += 1

        return c

    c_1 = _extract_coeffs(coeffs_1)
    c_2 = _extract_coeffs(coeffs_2)

    def _apply_pol(x, y):
        return np.stack([np.polynomial.polynomial.polyval2d(x, y, c_1),
                         np.polynomial.polynomial.polyval2d(x, y, c_2)])

    return _apply_pol


def read_list(f):
    if isinstance(f, str) or isinstance(f, pathlib.Path):
        with open(f, 'r') as fi:
            header, df, _, _ = read_list_ext(fi)

    else:
        header, df, _, _ = read_list_ext(f)

    return header, df


def read_list_ext(f):
    # Extract global @ parameters
    header = {}
    line = f.readline()
    curline = 1
    while line[0] == "@":
        line = line[:-1]
        splitted = line.split(" ")
        key = splitted[0][1:]

        try:
            value = list(map(float, splitted[1:]))

            if len(value) == 1:
                if str(int(splitted)) == splitted[1:]:
                    value = int(value[0])
                else:
                    value = value[0]
            else:
                value = pd.array(value)
        except:
            value = splitted[1:][0]

        header[key.lower()] = value

        line = f.readline()
        curline += 1

    # Extract dataframe
    # First, column names
    columns = []
    df_format = None
    df_desc = {}
    while line[0] == "#":
        curline += 1
        line = line[1:-1].strip()

        splitted = line.split(" ")
        if splitted[0].strip() == "end":
            break

        if splitted[0].strip() == "format":
            df_format = str(" ".join(line.split(" ")[1:])).strip()
            line = f.readline()
            continue

        splitted = line.split(":")
        column = splitted[0].strip()
        columns.append(column)
        if len(splitted) > 1:
            df_desc[column] = splitted[1].strip()

        line = f.readline()

    df = pd.read_csv(f, sep=" ", names=columns, index_col=False, skipinitialspace=True)

    return header, df, df_desc, df_format


def write_list(filename, header, df, df_desc, df_format):
    with open(filename, 'w') as f:
        for key in header.keys():
            if isinstance(header[key], Iterable) and not isinstance(header[key], str):
                f.write("@{} {}\n".format(key.upper(), " ".join(map(str, header[key]))))
            else:
                f.write("@{} {}\n".format(key.upper(), str(header[key])))

        for column in df.columns:
            if column in df_desc.keys():
                f.write("#{} : {}\n".format(column, df_desc[column]))
            else:
                f.write("#{}".format(column))

        if df_format is not None:
            f.write("# format {}\n".format(df_format))

        f.write("# end\n")

        df.to_csv(f, sep=" ", index=False, header=False)


class ListTable:
    def __init__(self, header, df, df_desc, df_format, filename=None):
        self.header = header
        self.df = df
        self.df_desc = df_desc
        self.df_format = df_format
        self.filename = filename

    @classmethod
    def from_filename(cls, filename):
        with open(filename, 'r') as f:
            header, df, df_desc, df_format = read_list_ext(f)

        return cls(header, df, df_desc, df_format, filename=filename)

    def write_to(self, filename):
        write_list(filename, self.header, self.df, self.df_desc, self.df_format)

    def write(self):
        self.write_to(self.filename)
