#!/usr/bin/env python3

import pathlib

import numpy as np
from croaks.match import NearestNeighAssoc
import pandas as pd


def cat_to_ds9regions(cat, filename, radius=5., color='green'):
    with open(filename, 'w') as f:
        f.write("global color={} dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n".format(color))
        f.write("image\n")
        for c in cat:
            f.write("circle {} {} 5\n".format(c['x'], c['y']))


def match_pixel_space(refcat, cat, radius=1.):
    def _euclidian(x1, y1, x2, y2):
        return np.sqrt((np.array(x1) - np.array(x2)) ** 2 + (np.array(y1) - np.array(y2)) ** 2)
    assoc =  NearestNeighAssoc(first=[refcat['x'], refcat['y']], radius=radius)
    index = assoc.match(cat['x'], cat['y'], metric=_euclidian)
    return index


def get_cat_size(catalog_filename):
    with open(catalog_filename, 'r') as f:
        _, cat = read_list(f)

    return len(cat)

def sc_ra(skycoord):
    return skycoord.frame.data.lon.value


def sc_dec(skycoord):
    return skycoord.frame.data.lat.value


def contained_in_exposure(objects, wcs):
    width, height = wcs.pixel_shape

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

    return objects[(sc_ra(objects) < tl[0]) & (sc_ra(objects) > br[0]) & (sc_dec(objects) > tl[1]) & (sc_dec(objects) < br[1])]


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
            return pathlib.Path(reference_quadrant[:-1])


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
    # Extract global @ parameters
    global_params = {}
    line = f.readline()
    curline = 1
    while line[0] == "@":
        line = line[:-1]
        splitted = line.split(" ")
        key = splitted[0][1:]

        try:
            value = list(map(float, splitted[1:]))

            if len(value) == 1:
                value = value[0]
            else:
                value = pd.array(value)
        except:
            value = splitted[1:][0]

        global_params[key.lower()] = value

        line = f.readline()
        curline += 1

    # Extract dataframe
    # First, column names
    columns = []
    while line[0] == "#":
        curline += 1
        line = line[1:-1].strip()

        splitted = line.split(" ")
        if splitted[0].strip() == "end":
            line = f.readline()
            break

        if splitted[0].strip() == "format":
            line = f.readline()
            continue

        splitted = line.split(":")
        columns.append(splitted[0].strip())

        line = f.readline()

    df = pd.read_csv(f, sep=" ", names=columns, index_col=False, skipinitialspace=True)

    return global_params, df