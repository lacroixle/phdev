#!/usr/bin/env python3

import pandas as pd
import numpy as np


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
