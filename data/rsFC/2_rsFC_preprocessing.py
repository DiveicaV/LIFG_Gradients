#!/usr/bin/env python
# coding: utf-8

# Script adapted by Veronica Diveica based on the following scripts written by Michael Riedel:
# https://github.com/NBCLab/niconn/blob/master/connectivity/rs_corr.py 
# https://github.com/NBCLab/niconn-hcp/blob/main/rs_preprocess-corr.py

# setup
from glob import glob
import os
import os.path as op
import pandas as pd
import nibabel as nib
from niflow.nipype1.workflows.fmri.fsl import create_susan_smooth 
from nipype.interfaces.fsl.model import GLM  
from nipype.interfaces import fsl as fsl
from nipype.pipeline import engine as pe 
from nipype.interfaces import utility
import nipype.interfaces.io as nio
import shutil # pip install pytest-shutil 
from nilearn.image import clean_img

# GSR & smoothing function
def rs_preprocess(in_file, fwhm, work_dir, output_dir):

    # define nodes and workflows
    rs_preproc_workflow = pe.Workflow(name="rs_preproc_workflow", base_dir=work_dir) # Controls the setup and execution of a pipeline of processes. (class nipype.pipeline.engine.workflows.Workflow(name, base_dir=None)

    # input
    inputnode = pe.Node(utility.IdentityInterface(fields=['func', 'fwhm']), name='inputspec')
    inputnode.inputs.func = in_file # specifies the input file functional image
    inputnode.inputs.fwhm = fwhm # specifies the input value for the smoothing kernel

    # make a brain mask
    immask = pe.Node(fsl.ImageMaths(op_string = '-abs -bin -Tmin'), name='immask') # absolute value, binarize, min across time
    rs_preproc_workflow.connect(inputnode, 'func', immask, 'in_file')

    # get time-series for GSR
    meants = pe.Node(fsl.utils.ImageMeants(), name='meants') # compute average timeseries across all voxels in mask
    rs_preproc_workflow.connect(inputnode, 'func', meants, 'in_file')
    rs_preproc_workflow.connect(immask, 'out_file', meants, 'mask')

    # global signal regression
    glm = pe.Node(GLM(), name='glm')
    glm.inputs.out_res_name = op.join(work_dir, 'res4d.nii.gz') # output file for residuals
    rs_preproc_workflow.connect(inputnode, 'func', glm, 'in_file')
    rs_preproc_workflow.connect(immask, 'out_file', glm, 'mask')
    rs_preproc_workflow.connect(meants, 'out_file', glm, 'design')

    # smoothing
    smooth = create_susan_smooth() # Smooth using FSL's SUSAN with the brightness threshold set to 75% of the median value of the image and a mask consituting the mean functional
    rs_preproc_workflow.connect(glm, 'out_res', smooth, 'inputnode.in_files')
    rs_preproc_workflow.connect(inputnode, 'fwhm', smooth, 'inputnode.fwhm')
    rs_preproc_workflow.connect(immask, 'out_file', smooth, 'inputnode.mask_file')
    
    # retrieve output
    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = work_dir
    rs_preproc_workflow.connect(glm, 'out_res', datasink, 'gsr')
    rs_preproc_workflow.connect(immask, 'out_file', datasink, 'mask')
    rs_preproc_workflow.connect(smooth.get_node('smooth'), 'smoothed_file', datasink, 'gsr_smooth') # WHAT IS THE PICK FIRST? # seemed to work the same without it for one participant

    rs_preproc_workflow.run()

    # rename & copy data to output directory
    gsr_fn = glob(op.join(work_dir, 'gsr', '*.nii.gz'))[0]
    mask_fn = glob(op.join(work_dir, 'mask', '*.nii.gz'))[0]
    gsr_smooth_fn = glob(op.join(work_dir, 'gsr_smooth', '*', '*.nii.gz'))[0]
    gsr_fn2 = op.join(output_dir, '{0}.nii.gz'.format(op.basename(in_file).split('.')[0]))
    mask_fn2 = op.join(output_dir, '{0}_mask.nii.gz'.format(op.basename(in_file).split('.')[0]))
    gsr_smooth_fn2 = op.join(output_dir, '{0}_smooth.nii.gz'.format(op.basename(in_file).split('.')[0]))

    shutil.copyfile(gsr_fn, gsr_fn2)
    shutil.copyfile(mask_fn, mask_fn2)
    shutil.copyfile(gsr_smooth_fn, gsr_smooth_fn2)

    shutil.rmtree(work_dir)


# function to pre-process participant-level data
def main(rs_data_dir, work_dir, ppt):

    nii_files = sorted(glob(op.join(rs_data_dir, ppt, 'rfMRI_REST*', 'rfMRI_REST*_hp2000_clean.nii.gz')))
    nii_files = [x for x in nii_files if '7T' not in x]

    for nii_fn in nii_files:
    
    	# specify participant path
        tmp_output_dir = op.join(rs_data_dir, 'derivatives', 'gsr+smooth', ppt) 
        if not op.isdir(tmp_output_dir): # create participant folder for pre-processed data if it does not exist
            os.makedirs(tmp_output_dir)
        
        # apply pre-processing pipeline including global signal regression and smoothing
        nii_work_dir = op.join(work_dir, 'rsfc', ppt, op.basename(nii_fn).split('.')[0]) # specify working directory
        rs_preprocess(nii_fn, 4, nii_work_dir, tmp_output_dir)
		
		# apply band-pass filter
		nii_img = op.join(tmp_output_dir, 'rfMRI_REST*_hp2000_clean_smooth.nii.gz')
        filtered = clean_img(nii_img, low_pass=0.08, high_pass=0.01, t_r=0.72, detrend=False, standardize=False)
        nib.save(filtered, op.join(tmp_output_dir, '{0}_filtered.nii.gz'.format(op.basename(nii_fn).split('.')[0])))
        
        
# specify directories
hcp_dir = os.getcwd()
data_dir = op.join(hcp_dir, 'hcp-openaccess', 'HCP1200')

# get participant IDs
pids_df = pd.read_csv(op.join(hcp_dir, 'hcp1200_participants-150.tsv'), sep='\t')
pids_df = pids_df['participant_id'].tolist()
pids_df = [str(item) for item in pids_df]

# Pre-process rsFC runs for all 150 participants 
for tmp_pid in pids_df:
	main(data_dir, hcp_dir, tmp_pid)
        
