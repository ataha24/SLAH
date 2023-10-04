#!/usr/bin/env python
# coding: utf-8

import numpy as np
import nibabel as nib


    
input_im = snakemake.input["im"]
output_im = snakemake.output["im_norm"]


def read_nii_metadata(nii_path):
    """Load in nifti data and header information"""
    nii = nib.load(nii_path)
    nii_affine = nii.affine
    nii_data = nii.get_fdata()
    #added normalization 
    nii_data = (nii_data - nii_data.min())/ (nii_data.max() - nii_data.min())

    return nii_affine, nii_data

intermediate = read_nii_metadata(input_im)


nib.save(nib.Nifti1Image(intermediate[1], affine=intermediate[0]), output_im)

