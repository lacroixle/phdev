#!/usr/bin/env python3


def reference_quadrant(band_path, ztfname, filtercode, logger, args):
    """

    """
    # TODO: Rewrite this using quadrants.parquet instead of reading all fits headers
    import pathlib
    import pandas as pd
    import numpy as np
    from astropy.wcs import WCS
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    from utils import get_header_from_quadrant_path, read_list

    # Determination of the best seeing quadrant
    # First determine the most represented field
    logger.info("Determining best seeing quadrant...")
    quadrant_paths = [folder for folder in band_path.glob("ztf_*".format(ztfname, filtercode)) if folder.is_dir()]
    quadrant_paths = list(filter(lambda x: x.joinpath("psfstars.list").exists(), quadrant_paths))

    seeings = {}
    for quadrant_path in quadrant_paths:
        quadrant_header = get_header_from_quadrant_path(quadrant_path)
        seeings[quadrant_path] = (quadrant_header['seseeing'], quadrant_header['fieldid'])

    fieldids = list(set([seeing[1] for seeing in seeings.values()]))
    fieldids_count = [sum([1 for f in seeings.values() if f[1]==fieldid]) for fieldid in fieldids]
    maxcount_field = fieldids[np.argmax(fieldids_count)]

    logger.info("{} different field ids".format(len(fieldids)))
    logger.info("Field ids: {}".format(fieldids))
    logger.info("Count: {}".format(fieldids_count))
    logger.info("Max quadrant field={}".format(maxcount_field))

    seeing_df = pd.DataFrame([[quadrant, seeings[quadrant][0]] for quadrant in seeings.keys() if seeings[quadrant][1]==maxcount_field], columns=['quadrant', 'seeing'])
    seeing_df = seeing_df.set_index(['quadrant'])

    # Remove exposure where small amounts of stars are detected
    seeing_df['n_standalonestars'] = list(map(lambda x: len(read_list(pathlib.Path(x).joinpath("standalone_stars.list"))[1]), seeing_df.index))
    seeing_df = seeing_df.loc[seeing_df['n_standalonestars'] >= 25]

    idxmin = seeing_df.idxmin().values[0]
    minseeing = seeing_df.at[idxmin, 'seeing']

    logger.info("Best seeing quadrant: {}". format(idxmin))
    logger.info("  with seeing={}".format(minseeing))

    logger.info("Reading SN1a parameters")
    sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='sn_info')

    logger.info("Reading reference WCS")
    with fits.open(pathlib.Path(idxmin).joinpath("calibrated.fits")) as hdul:
        w = WCS(hdul[0].header)

    ra_px, dec_px = w.world_to_pixel(SkyCoord(ra=sn_parameters['sn_ra'], dec=sn_parameters['sn_dec'], unit='deg'))

    logger.info("Writing driver file")
    driver_path = band_path.joinpath("{}_driver_{}".format(ztfname, filtercode))
    logger.info("Writing driver file at location {}".format(driver_path))
    with open(driver_path, 'w') as f:
        f.write("OBJECTS\n")
        f.write("{} {} DATE_MIN={} DATE_MAX={} NAME={} TYPE=0 BAND={}\n".format(ra_px[0],
                                                                                dec_px[0],
                                                                                sn_parameters['t_inf'].values[0],
                                                                                sn_parameters['t_sup'].values[0],
                                                                                ztfname,
                                                                                filtercode))
        f.write("IMAGES\n")
        for quadrant_path in quadrant_paths:
            f.write("{}\n".format(quadrant_path))
        f.write("PHOREF\n")
        f.write("{}\n".format(idxmin))
        f.write("PMLIST\n")
        f.write(str(band_path.joinpath("astrometry/pmcatalog.list")))

    with open(band_path.joinpath("reference_quadrant"), 'w') as f:
        f.write(str(idxmin.name))


def smphot(band_path, ztfname, filtercode, logger, args):
    from deppol_utils import run_and_log

    logger.info("Running scene modeling")

    smphot_output = band_path.joinpath("smphot_output")
    smphot_output.mkdir(exist_ok=True)

    run_and_log(["mklc", "-t", band_path.joinpath("mappings"), "-O", smphot_output, "-v", band_path.joinpath("{}_driver_{}".format(ztfname, filtercode))], logger=logger)

    return True


def smphot_plot(band_path, ztfname, filtercode, logger, args):
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # logger.info("Running pmfit plots")
    # driver_path = band_path.joinpath("{}_driver_{}".format(ztfname, filtercode))
    # gaia_path = band_path.joinpath("{}/{}/gaia.npy".format(ztfname, filtercode))
    # run_and_log(["pmfit", driver_path, "--gaia={}".format(gaia_path), "--outdir={}".format(cwd.joinpath("pmfit")), "--plot-dir={}".format(cwd.joinpath("pmfit_plot")), "--plot", "--mu-max=20."], logger=logger)

    # logger.info("Running smphot plots")
    # with open(band_path.joinpath("smphot_output/lightcurve_sn.dat"), 'r') as f:
    #     _, sn_flux_df = list_format.read_list(f)

    # plt.errorbar(sn_flux_df['mjd'], sn_flux_df['flux'], yerr=sn_flux_df['varflux'], fmt='.k')
    # plt.xlabel("MJD")
    # plt.ylabel("Flux")
    # plt.title("Calibrated lightcurve - {} - {}".format(ztfname, filtercode))
    # plt.grid()
    # plt.savefig(band_path.joinpath("{}-{}_smphot_lightcurve.png".format(ztfname, filtercode)), dpi=300)
    # plt.close()

    return True
