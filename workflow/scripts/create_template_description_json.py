import nibabel as nib
import numpy as np
import json


img = nib.load(snakemake.input[0])


hdr = img.header
shape = np.array(hdr.get_data_shape()).astype('float').tolist()
zooms = np.array(hdr.get_zooms()).astype('float').tolist()

affine = img.affine
origin = affine @ np.array([0,0,0,1]).T
origin = origin[0:3].astype('float').tolist()

template_dict = dict()

#add extras from config file
template_dict.update(snakemake.config['template_description_extras'])

#add shape, zooms, origin, for the resolution
template_dict.update( { 'res':  {'{res:02d}'.format(res=snakemake.config['resolution_index']) : { 'origin': origin, 'shape': shape, 'zooms': zooms } } } )

#add cohorts to json, with full list of subjects for each cohort..
cohort_dict = dict()
for cohort in snakemake.params.cohorts:
    cohort_dict[cohort] = { 'participants': snakemake.params.subjects[cohort] } 

template_dict.update( { 'cohort': cohort_dict } )


with open(snakemake.output[0],'w') as outfile:
    json.dump(template_dict,outfile,indent=2)

