#!/usr/bin/env python
# coding: utf-8

# script written by Michael Riedel and Veronica Diveica

# setup
import os
import pickle
import numpy as np
import os.path as op
import nibabel as nib
from nimare.extract import fetch_neuroquery
from nimare.io import convert_neurosynth_to_dataset
from nimare.extract import download_abstracts
from nimare.dataset import Dataset
from nimare.utils import vox2mm
from nimare.meta.cbma.ale import ALE
from nilearn.masking import unmask
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import load_img, math_img
from nilearn import plotting
from brainspace.gradient import GradientMaps
import matplotlib.pyplot as plt

#establish project direcotry
project_dir = os.getcwd()

# output directory
out_dir = op.join(project_dir, 'meta-analytic_gradients')
os.makedirs(out_dir, exist_ok=True)

#sub-directories 
os.makedirs(op.join(out_dir, 'corrmats'), exist_ok=True)
os.makedirs(op.join(out_dir, 'maps'), exist_ok=True)
os.makedirs(op.join(out_dir, 'gradient_explore'), exist_ok=True)

#data direcotry
data_dir = op.abspath(op.join(project_dir, 'data'))

#load ROI
roi = op.join(data_dir,'Left_IFG_ROI.nii.gz')
roi_img = nib.load(roi)
roi_idx = np.vstack(np.where(roi_img.get_fdata())).T # get list of indices for the non-zero (i.e. inside roi) voxels 
roi_coords = vox2mm(roi_idx, roi_img.affine) # convert image space locations (ijk) from voxel to mm space

# Define function to download NeuroQuery data, taken from https://nimare.readthedocs.io/en/latest/generated/nimare.extract.fetch_neuroquery.html?highlight=download_neuroquery#nimare.extract.fetch_neuroquery
# Neuroquery data repository available at: https://github.com/neuroquery/neuroquery_data/tree/master/data
def download_neuroquery(data_dir):
    files = fetch_neuroquery(
        data_dir=data_dir,
        version="1",
        overwrite=False,
        source="combined",
        vocab="neuroquery6308",
        type="tfidf",
    ) # Note that the files are saved to a new folder within "data_dir" named "neuroquery".

    # convert Neuroquery dataset to nimare dataset
    # Function source https://nimare.readthedocs.io/en/latest/generated/nimare.io.convert_neurosynth_to_dataset.html#nimare.io.convert_neurosynth_to_dataset
    neuroquery_db = files[0] # create dictionary to store dataset
    dset = convert_neurosynth_to_dataset(
        coordinates_file=neuroquery_db["coordinates"],
        metadata_file=neuroquery_db["metadata"],
        annotations_files=neuroquery_db["features"],
    )
    dset.save(op.join(data_dir, "neuroquery_dataset.pkl.gz"))
    dset = download_abstracts(dset, "psuda2@bangor.ac.uk") #Download the abstracts for a list of PubMed IDs.
    dset.save(op.join(data_dir, "neuroquery_dataset_with_abstracts.pkl.gz")) 

## STEP 1 
# compute correlation matrix if corrmat not already in directory
if not op.isfile(op.join(out_dir, 'corrmat.pkl')):

    #location of the neuroquery dataset
    neuroquery_dset = op.join(data_dir, 'neuroquery', 'neuroquery_dataset_with_abstracts.pkl.gz')
    # download dataset if dataset not already in directory
    if not op.isfile(neuroquery_dset):
        download_neuroquery(op.join(data_dir, 'neuroquery'))
    else:
        dset = Dataset.load(neuroquery_dset)
    #generate an ALE image for each coordinate by running ALE algorithm on studies reporting foci within 6mm of the coordinate of interest
    macm_ales = np.zeros((roi_coords.shape[0], 228453)) # creates array filled with zeros with no. of rows = no. of roi voxels and no. of columns = 228453 (whole brain)
    for i_coord in range(roi_coords.shape[0]):
        coord = roi_coords[i_coord, :][None, :] # extract coordonates x, y, z
        coord_ids = dset.get_studies_by_coordinate(coord, r=6) # Extract list of studies with at least one peak within radius of coordonate
        coord_dset = dset.slice(coord_ids) # Extract data from studies with at least one peak within radius of coordonate
        # compute ALE map across all identified neuroquery studies (within 6 mm of each coord)
        ale = ALE(kernel__fwhm=15)
        images = ale.fit(coord_dset)
        # save roi coord x whole-brain ALE values (MACM) in array
        macm_ales[i_coord,:] = images.maps['stat']

    #save roi coord x whole-brain ALE values (MACM) matrix
    with open(op.join(out_dir, 'MACM_ALEs.pkl'), 'wb') as fo: # opens file for writing in binary mode
        pickle.dump(macm_ales, fo, protocol=pickle.HIGHEST_PROTOCOL)
        
    # apply correlation to roi coord x whole-brain ALE values (MACM) matrix 
    correlation = ConnectivityMeasure(kind='correlation')
    macm_ales_correlation_matrix = correlation.fit_transform([np.transpose(macm_ales)])[0]
    #save coactivation correlation matrix
    with open(op.join(out_dir, 'corrmat.pkl'), 'wb') as fo: # opens file for writing in binary mode
        pickle.dump(macm_ales_correlation_matrix, fo, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(op.join(out_dir, 'corrmat.pkl'), 'rb') as fo: # if corrmat exists in directory, load corrmat
        macm_ales_correlation_matrix = pickle.load(fo)

# visualize the correlation matrix
fig2, ax2 = plt.subplots(figsize=(8,8), dpi=150)     
c2 = ax2.matshow(macm_ales_correlation_matrix, cmap ='jet', vmin = 0.5)
fig2.colorbar(c2, ax = ax2)
plt.title('MACM Similarity Matrix - unsorted & thresholded')
plt.savefig(op.join(out_dir, 'corrmat.png'))
plt.close()


## STEP 2
# Build gradients using diffusion embedding & strongest 10% of connections
gm = GradientMaps(n_components=10, approach='dm', kernel='cosine')
gm.fit(macm_ales_correlation_matrix, sparsity=0.9)

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
plt.show()

# save gradient maps & sorted corrmats
for i_grad in range(gm.gradients_.shape[1]):
    # save the gradient map for each of 10 gradients 
    tmp_grad_img = unmask(gm.gradients_[:,i_grad], roi_img) # take the masked data and bring it back to 3D
    nib.save(tmp_grad_img, op.join(out_dir,'maps','gradient-{}.nii.gz'.format(i_grad)))
    
    plotting.plot_stat_map(op.join(out_dir,'maps', 'gradient-{}.nii.gz'.format(i_grad)), draw_cross=False, dim=-0.3, cmap=plt.cm.gist_rainbow, display_mode='mosaic')
    plt.savefig(op.join(out_dir,'maps', 'gradient-{}_mosaic.png'.format(i_grad)))
    plotting.plot_stat_map(op.join(out_dir,'maps', 'gradient-{}.nii.gz'.format(i_grad)), draw_cross=False, dim=-0.3, cmap=plt.cm.gist_rainbow)
    plt.savefig(op.join(out_dir,'maps', 'gradient-{}.png'.format(i_grad)))
    
    # save sorted correlation matrix for each gradient
    grad_sort_inds = np.argsort(gm.gradients_[:,i_grad])
    grad_sort_corr_mat = macm_ales_correlation_matrix.copy()
    grad_sort_corr_mat = grad_sort_corr_mat[grad_sort_inds,:]
    grad_sort_corr_mat = grad_sort_corr_mat[:,grad_sort_inds]
    
    # plot sorted correlation matrix for each gradient
    fig, ax = plt.subplots(figsize=(8,8), dpi=150)     
    c = ax.matshow(grad_sort_corr_mat, cmap = 'jet', vmin = 0.5) 
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
# estimate the algebraic connectivity of the matrix
# Build gradients using Laplacian eigenmaps & strongest 10% of connections
gm = GradientMaps(n_components=10, approach='le', kernel='cosine')
gm.fit(macm_ales_correlation_matrix, sparsity=0.9)

#save lambdas
with open(op.join(out_dir, 'laplacian_lambdas.txt'), 'w') as fo:
    np.savetxt(fo, gm.lambdas_)
    