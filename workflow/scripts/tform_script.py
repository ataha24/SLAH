#!/usr/bin/env python
# coding: utf-8

import csv
import re

import numpy as np
import pandas as pd


def determineFCSVCoordSystem(input_fcsv):
    # need to determine if file is in RAS or LPS
    # loop through header to find coordinate system
    coordFlag = re.compile("# CoordinateSystem")
    coord_sys = None
    with open(input_fcsv, "r+") as fid:
        rdr = csv.DictReader(filter(lambda row: row[0] == "#", fid))
        row_cnt = 0
        for row in rdr:
            cleaned_dict = {k: v for k, v in row.items() if k is not None}
            if any(coordFlag.match(x) for x in list(cleaned_dict.values())):
                coordString = list(filter(coordFlag.match, list(cleaned_dict.values())))
                assert len(coordString) == 1
                coord_sys = coordString[0].split("=")[-1].strip()
            row_cnt += 1
    return coord_sys


fcsv_source = snakemake.input["groundtruth"]
xfm_txt = snakemake.input["xfm_new"]
template = snakemake.params["template"]
fcsv_new = snakemake.output["fcsv_new"]

# load transform from subj to template
sub2template = np.loadtxt(xfm_txt)
fcsv_df = pd.read_table(fcsv_source, sep=",", header=2)

# if LPS coordinate system, change to RAS
coordSys = determineFCSVCoordSystem(fcsv_source)
if any(x in coordSys for x in {"LPS", "1"}):
    fcsv_df["x"] = -1 * fcsv_df["x"]  # flip orientation in x
    fcsv_df["y"] = -1 * fcsv_df["y"]  # flip orientation in y

coords = fcsv_df[["x", "y", "z"]].to_numpy()

# to plot in mni space, need to transform coords
tcoords = np.zeros(coords.shape)
for i in range(len(coords)):
    vec = np.hstack([coords[i, :], 1])
    tvec = np.linalg.inv(sub2template) @ vec.T
    tcoords[i, :] = tvec[:3]

with open(template, "r", encoding="utf-8") as file:
    list_of_lists = []
    reader = csv.reader(file)
    for i in range(3):
        list_of_lists.append(next(reader))
    for idx, val in enumerate(reader):
        val[1:4] = tcoords[idx][:3]
        list_of_lists.append(val)

with open(fcsv_new, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(list_of_lists)
