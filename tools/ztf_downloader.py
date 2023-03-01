#!/usr/bin/env python3

import argparse
import pathlib
import time
import traceback
import gc

from itertools import chain
import numpy as np
from ztfimg import science
from ztfquery import io
from joblib import Parallel, delayed
from more_itertools import chunked
# from dask import delayed, compute
# from dask.distributed import Client, LocalCluster, wait, get_worker

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--quadrant-list', type=pathlib.Path)
    argparser.add_argument('--chunk-size', type=int, default=1000)
    argparser.add_argument('-j', type=int)
    argparser.add_argument('--error-filename', type=pathlib.Path)

    args = argparser.parse_args()
    args.quadrant_list = args.quadrant_list.expanduser().resolve()
    args.error_filename = args.error_filename.expanduser().resolve()

    print("Reading quadrant list from file {}".format(args.quadrant_list))
    quadrant_list = []
    with open(args.quadrant_list, 'r') as f:
        quadrant_list.extend(map(lambda x: x.strip(), f.readlines()))

    quadrant_chunks = list(chunked(quadrant_list, args.chunk_size))
    print("{} chunks of {} quadrants to download".format(len(quadrant_chunks), args.chunk_size), flush=True)

    # print("Allocating local cluster")
    # localCluster = LocalCluster(n_workers=1, dashboard_address='localhost:8787', processes=True, threads_per_worker=1, nanny=True)
    # client = Client(localCluster)
    # print("Allocating {} workers".format(args.j))
    # localCluster.scale(args.j)
    # print("Done. Dashboard address={}".format(client.dashboard_link))

    # Wipe error list file
    with open(args.error_filename, 'w') as f:
        pass

    def _download_quadrant(quadrant):
        warnings.filterwarnings("ignore")
        try:
            print(".", end="", flush=True)
            if "o.fits.fz" in quadrant:
                downloaded_raw_ccd = io.get_file(quadrant)
                # print(downloaded_raw_ccd)
            else:
                downloaded_quadrant_sciimg = io.get_file(quadrant)
                downloaded_quadrant_mskimg = io.get_file(quadrant, suffix="mskimg.fits")
            gc.collect()
        except PermissionError as e:
            traceback.print_exc()
            return False
        finally:
            return True

    def _download_quadrant_chunk(quadrant_chunk):
        return Parallel(n_jobs=args.j)(delayed(_download_quadrant)(quadrant) for quadrant in quadrant_chunk)
        #download_jobs = [delayed(_download_quadrant)(quadrant) for quadrant in quadrant_chunk]
        #gc.collect()
        #fjobs = client.compute(download_jobs)
        #return [fjob.result() for fjob in fjobs]


    chunk_size_gb = args.chunk_size*(37+17)/1000
    for i, quadrant_chunk in enumerate(quadrant_chunks):
        print("Chunk {}, {} quadrants to download ({} GB)".format(i, len(quadrant_chunk), chunk_size_gb), flush=True)
        start_time = time.perf_counter()
        success = _download_quadrant_chunk(quadrant_chunk)
        elapsed_time = time.perf_counter() - start_time
        quadrants_error = np.array(quadrant_chunk)[not success].tolist()
        with open(args.error_filename, 'a') as f:
            f.writelines([line + "\n" for line in quadrants_error])

        gc.collect()

        print("\tSuccessfuly downloaded {} quadrants".format(sum(success)))
        print("\tElapsed time={} s, transfer rate={} MB/s".format(elapsed_time, chunk_size_gb*1000/elapsed_time))

    print("Done", flush=True)

