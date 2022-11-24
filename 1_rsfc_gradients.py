#!/usr/bin/env python
# coding: utf-8

# script written by Michael Riedel and Veronica Diveica

# setup
import os
import pickle
import numpy as np
import os.path as op
import nibabel as nib
import pandas as pd
from glob import glob
from nilearn.masking import unmask
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiMasker
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps
import matplotlib.pyplot as plt
from nilearn import plotting

# establish project direcotry
project_dir = os.getcwd()

# output directory
out_dir = op.join(project_dir, 'rsFC_gradients')
os.makedirs(out_dir, exist_ok=True)

# sub-directories 
os.makedirs(op.join(out_dir, 'corrmats'), exist_ok=True)
os.makedirs(op.join(out_dir, 'maps'), exist_ok=True)
os.makedirs(op.join(out_dir, 'laplacian'), exist_ok=True)

# load ROI
roi = op.join(project_dir, 'data/Left_IFG_ROI.nii.gz')
roi_img = nib.load(roi)
roi_img_mask = NiftiMasker(mask_img=roi_img).fit() # create Nifti Masker object


## STEP 1 
# compute correlation matrix if corrmat not already in directory
if not op.isfile(op.join(out_dir, 'corrmat.pkl')):
    # get paths to participant data
    hcp_data_dir = op.join(project_dir, 'data/rsFC/hcp-openaccess/HCP1200/derivatives/gsr+smooth'
    hcp_subs = os.listdir(hcp_data_dir)
    hcp_subs.remove('.DS_Store')
	# compute similarity matrix
    correlation = ConnectivityMeasure(kind='correlation') # define similarity metric
    corrmat = np.zeros((len(np.where(roi_img.get_fdata())[0]), len(np.where(roi_img.get_fdata())[0]))) # generate empty group matrix
    # compute correlation matrices for all participants
    for j, sub in enumerate(hcp_subs): # get subject IDs
        print(j) # print iteration no. 
        # compute participant correlation matrix if it does not already exist
        if not op.isfile(op.join(out_dir,'subject_level', sub, 'corrmat.pkl')):
            runs = glob(op.join(hcp_data_dir, sub, '*clean_smooth_filtered.nii.gz')) # identify all preprocessed rs-fMRI images in sub folder
            sub_mean_corrmat = np.zeros((len(np.where(roi_img.get_fdata())[0]), len(np.where(roi_img.get_fdata())[0]))) # generate empty participant matrix
            for i, run in enumerate(runs):
                roi_ts = roi_img_mask.transform(run) # 2D matrix n_time_points x n_voxels
                roi_ts_na_idx = np.where(np.any(roi_ts, axis=0) == False)[0] # identifies voxels with no associated timeseries
                roi_ts = np.delete(roi_ts, roi_ts_na_idx, axis=1) # deletes data column for voxels with no associated timeseries
                tmp_sub_corrmat = correlation.fit_transform([roi_ts])[0] # compute cross-correlation matrix n_voxels x n_voxels for run
                #transform to z-scores
                tmp_sub_corrmat = np.arctanh(tmp_sub_corrmat) # apply Fisher's r-to-Z transformation
                tmp_sub_corrmat = np.insert(tmp_sub_corrmat, np.subtract(roi_ts_na_idx, np.arange(len(roi_ts_na_idx))), 0, axis=0) # insert back column data for voxels with no associaed timeseries 
                tmp_sub_corrmat = np.insert(tmp_sub_corrmat, np.subtract(roi_ts_na_idx, np.arange(len(roi_ts_na_idx))), 0, axis=1) # insert back row data for voxels with no associaed timeseries
                np.fill_diagonal(tmp_sub_corrmat, np.arctanh(1-np.finfo(float).eps)) # insert max value on diagonal
                sub_mean_corrmat = sub_mean_corrmat + tmp_sub_corrmat # add together the matrices for each rs-fMRI run
			# divide by no. of runs to compute mean participant correlation matrix
            sub_mean_corrmat = sub_mean_corrmat/(i+1) 
            #save participant correlation matrix
            os.makedirs(op.join(out_dir, 'subject_level', sub), exist_ok=True)
            with open(op.join(out_dir,'subject_level', sub, 'corrmat.pkl'), 'wb') as fo:
                pickle.dump(sub_mean_corrmat, fo, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(op.join(out_dir,'subject_level', sub, 'corrmat.pkl'), 'rb') as fo:
                sub_mean_corrmat = pickle.load(fo)
		# add participant's correlation matrix to the group matrix
        corrmat = corrmat + sub_mean_corrmat
	# divide by no. of subjects to get mean correlation matrix across subjects
    corrmat = corrmat/(j+1) 
    #save correlation matrix
    with open(op.join(out_dir, 'corrmat.pkl'), 'wb') as fo:
        pickle.dump(corrmat, fo, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(op.join(out_dir, 'corrmat.pkl'), 'rb') as fo:
        corrmat = pickle.load(fo)    
corrmat = np.tanh(corrmat) # transform the Z transformed correlation matrix back to r values

# visualize the group correlation matrix
fig, ax = plt.subplots(figsize=(8,8), dpi=150)      
c = ax.matshow(corrmat, cmap ='jet', vmin = -0.05, vmax=0.3)
fig.colorbar(c, ax = ax)
plt.savefig(op.join(out_dir, 'corrmat.png'))


## STEP 2
# Build gradients using diffusion embedding & strongest 10% of connections
gm = GradientMaps(n_components=10, approach='dm', kernel='cosine')
gm.fit(corrmat, sparsity=0.9)

# save gradients
with open(op.join(out_dir, 'gradients.pkl'), 'wb') as fo:
          pickle.dump(gm.gradients_, fo, protocol=pickle.HIGHEST_PROTOCOL)

#save lambdas
with open(op.join(out_dir, 'lambdas.txt'), 'w') as fo:
    np.savetxt(fo, gm.lambdas_)
    
# plot lambdas
lambdas = gm.lambdas_

fig, (ax) = plt.subplots(1,3, figsize=(20, 5), dpi = 150)

ax[0].plot(range(lambdas.size), lambdas, marker='o', linestyle='-')
ax[0].set_xlabel('Component No.', fontsize=16)
ax[0].set_ylabel('Eigenvalue', fontsize=16)
ax[0].set_title('Component Lambdas', fontsize=20)

vals = lambdas / np.sum(lambdas)
ax[1].plot(range(vals.size), vals, marker='o', linestyle='-')
ax[1].set_xlabel('Component No.', fontsize=16)
ax[1].set_ylabel('Variance', fontsize=16)
ax[1].set_title('Component Variance', fontsize=20)

vals = np.cumsum(lambdas) / np.sum(lambdas)
ax[2].plot(range(vals.size), vals, marker='o', linestyle='-')
ax[2].set_xlabel('Component No.', fontsize=16)
ax[2].set_ylabel('Cumulative Variance', fontsize=16)
ax[2].set_ylim(ymin=0)
ax[2].set_title('Component Cumulative Variance', fontsize=20)

plt.savefig(op.join(out_dir, 'lambdas.png'))

# save gradient maps & sorted corrmats
for i_grad in range(gm.gradients_.shape[1]):

    # save the gradient map for each of 10 gradients 
    tmp_grad_img = unmask(gm.gradients_[:,i_grad], roi_img) # take the masked data and bring it back to 3D
    nib.save(tmp_grad_img, op.join(out_dir,'maps','gradient-{}.nii.gz'.format(i_grad)))
    
    # plot gradient maps
    plotting.plot_stat_map(op.join(out_dir,'maps', 'gradient-{}.nii.gz'.format(i_grad)), draw_cross=False, dim=-0.3, cmap=plt.cm.gist_rainbow, display_mode='mosaic')
    plt.savefig(op.join(out_dir,'maps', 'gradient-{}_mosaic.png'.format(i_grad)))
    plotting.plot_stat_map(op.join(out_dir,'maps', 'gradient-{}.nii.gz'.format(i_grad)), draw_cross=False, dim=-0.3, cmap=plt.cm.gist_rainbow)
    plt.savefig(op.join(out_dir,'maps', 'gradient-{}.png'.format(i_grad)))
    
    # sort correlation matrix for each gradient
    grad_sort_inds = np.argsort(gm.gradients_[:,i_grad])
    grad_sort_corr_mat = corrmat.copy()
    grad_sort_corr_mat = grad_sort_corr_mat[grad_sort_inds,:]
    grad_sort_corr_mat = grad_sort_corr_mat[:,grad_sort_inds]
    
    # plot sorted correlation matrix for each gradient
    fig, ax = plt.subplots(figsize=(8,8), dpi=150)     
    c = ax.matshow(grad_sort_corr_mat, cmap = 'jet', vmin = -0.10, vmax=0.3) 
    fig.colorbar(c, ax = ax)
    plt.savefig(op.join(out_dir, 'corrmats', 'gradient-{}_corrmat-sorted.png'.format(i_grad)))

# rescale first two gradients for better visualisation
for grad_no in range(0,2):
    gradient = gm.gradients_[:,grad_no]
    gradient_rescaled = []
    min_v = abs(min(gradient)) + 1
    for i in gradient:
        gradient_rescaled.append(i + min_v)
    # save rescaled gradient map 
    grad_img = unmask(gradient_rescaled, roi_img) # take the masked data and bring it back to 3D
    nib.save(grad_img, op.join(out_dir, 'maps', 'gradient-{}_rescaled.nii.gz'.format(grad_no)))
    
# save voxels' gradient values for the first two gradients
with open(op.join(out_dir, 'gradient-0_voxel_gradient_values.csv'), 'w') as fo:
    np.savetxt(fo, gm.gradients_[:,0], fmt='%f')
with open(op.join(out_dir, 'gradient-1_voxel_gradient_values.csv'), 'w') as fo:
    np.savetxt(fo, gm.gradients_[:,1], fmt='%f')    

## STEP 3
# estimate the algebraic connectivity of the group matrix
# Build gradients using laplacian eigenmaps & strongest 10% of connections
gm = GradientMaps(n_components=10, approach='le', kernel='cosine')
gm.fit(corrmat, sparsity=0.9)
#save lambdas
with open(op.join(out_dir, 'laplacian/group_lambdas.txt'), 'w') as fo:
    np.savetxt(fo, gm.lambdas_)

# estimate the algebraic connectivity at the subject-level
sub_dir = op.join(out_dir, 'subject_level')
subs = os.listdir(sub_dir)
subs.remove('.DS_Store')
ac = [] # create dataframe to store algebraic connectivity values
for j, sub in enumerate(subs): # get subject IDs
    with open(op.join(out_dir,'subject_level', sub, 'corrmat.pkl'), 'rb') as fo:
        sub_mean_corrmat = pickle.load(fo) # load participant corramt
    sub_mean_corrmat = np.tanh(sub_mean_corrmat) # transform back to r values
    # laplacian eigenmaps
    gm = GradientMaps(n_components=10, approach='le', kernel='cosine')
    gm.fit(sub_mean_corrmat, sparsity=0.9)
    # save gradation metric (second largest eigenvalue)
    ac.append([sub, gm.lambdas_[9]])
# save participants' algebraic connectivity values
df = pd.DataFrame(ac, columns=["Subject", "Algebraic Connectivity"])
df.to_csv(op.join(out_dir, 'laplacian', 'Participant_level_algebraic_connectivity.csv'))
