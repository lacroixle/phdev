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

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--quadrant-list', type=pathlib.Path)
    argparser.add_argument('--chunk-size', type=int, default=1000)
    argparser.add_argument('-j', type=int)

    args = argparser.parse_args()
    args.quadrant_list = args.quadrant_list.expanduser().resolve()

    print("Reading quadrant list from file {}".format(args.quadrant_list))
    quadrant_list = []
    with open(args.quadrant_list, 'r') as f:
        quadrant_list.extend(map(lambda x: x.strip(), f.readlines()))

    quadrant_chunks = list(chunked(quadrant_list, args.chunk_size))
    print("{} chunks of {} quadrants to download".format(len(quadrant_chunks), args.chunk_size))

    quadrant_errors = []

    def _download_quadrant(quadrant):
        try:
            downloaded_quadrant_sciimg = io.get_file(quadrant)
            downloaded_quadrant_mskimg = io.get_file(quadrant, suffix="mskimg.fits")
        except PermissionError as e:
            quadrant_errors.append(quadrant)
            return False
        finally:
            return True

    def _download_quadrant_chunk(quadrant_chunk):
        print("Dowloading sciimg", flush=True)
        start_time = time.perf_counter()
        downloaded_quadrants = io.get_file(quadrant_chunk, maxnprocess=args.j)
        elapsed_time = time.perf_counter() - start_time
        print("Done. Dowloaded {} quadrants at MB/s".format(len(downloaded_quadrants), 37.*len(quadrant_chunk)/elapsed_time))
        print("Elapsed time={}".format(elapsed_time))

        print("Downloading mskimg", flush=True)
        start_time = time.perf_counter()
        downloaded_quadrants = io.get_file(quadrant_chunk, suffix="mskimg.fits", maxnprocess=args.j)
        elapsed_time = time.perf_counter() - start_time
        print("Done. Dowloaded {} quadrants at {} MB/s".format(len(downloaded_quadrants), 15.*len(quadrant_chunk)/elapsed_time))
        print("Elapsed time={}".format(elapsed_time))

        gc.collect()

    for i, quadrant_chunk in enumerate(quadrant_chunks):
        print("Chunk {}, {} quadrants to download ({} MB)".format(i, len(quadrant_chunk), ), flush=True)
        continue_download = True
        current_chunk = quadrant_chunk
        while continue_download:
            try:
                _download_quadrant_chunk(current_chunk)
            except PermissionError as e:
                print("Permission error for file {}".format(str(e)))
                to_remove = pathlib.Path(str(e)).with_suffix("").with_suffix("").with_suffix("").with_suffix(".fits").name
                to_remove = to_remove.replace("mskimg", "sciimg")
                print("Removing {} from chunk...".format(to_remove))
                current_chunk.remove(to_remove)
            except:
                traceback.print_exc()
                exit()
            else:
                continue_download = False


        print("Done downloading chunk {}".format(i))

    print("Done", flush=True)

