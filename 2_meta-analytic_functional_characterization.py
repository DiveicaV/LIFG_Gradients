
# script written by Veronica Diveica

import os
import os.path as op
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from nilearn import reporting, plotting
from nilearn.image import threshold_img, math_img, load_img, resample_to_img, index_img
from nimare.correct import FWECorrector
from nimare.dataset import Dataset
from nimare.meta import ALE
from nimare.meta.cbma.ale import ALESubtraction
from scipy.stats import norm
from nilearn.datasets import load_mni152_brain_mask


# set paths
project_dir = os.getcwd()
volume_dir = op.join(project_dir, 'meta-analytic_gradients', 'gradient_explore', 'volumes')
os.makedirs(volume_dir, exist_ok=True)
out_dir = op.join(project_dir, 'meta-analytic_gradients', 'gradient_explore', 'macms')
os.makedirs(out_dir, exist_ok=True)

# load ROI
roi_img = load_img(op.join(project_dir, 'data', 'Left_IFG_ROI.nii.gz'))


## STEP 1
# extract clusters based on gradient values

# load gradient values matrix
with open(op.join(project_dir, 'meta-analytic_gradients', 'gradients.pkl'), 'rb') as fo:
    gradients = pickle.load(fo)

# extract clusters based on the first two gradients
for gradient_no in range(0,2):
	gradient_img = load_img(op.join(project_dir, 'meta-analytic_gradients', 'maps', 'gradient-{gradient_no}.nii.gz')) # laod gradient map
	# get threshsold values
	gradient = gradients[:,0] 
	pct = list(range(0, 110, 20))
	thresh = np.percentile(gradient, pct)
	# threshold the gradient map
	for i in range(0, 5):
    	thresh_min = thresh[i]
    	thresh_max = thresh[i+1]
    	if(np.sign(thresh_min) == np.sign(thresh_max)):
        	cluster1 = math_img(f'img >= {thresh_min}', img=gradient_img)
        	cluster2 = math_img(f'img <= {thresh_max}', img=gradient_img)
        	cluster = math_img("img1 & img2", img1=cluster1, img2=cluster2)
    	else:
        	cluster1 = math_img(f'np.logical_not(img <= {thresh_max})', img=gradient_img)
        	cluster2 = math_img(f'np.logical_not(img >= {thresh_min})', img=gradient_img)
        	cluster = math_img(f'img1 - img2 - img3', img1 = roi_img, img2 = cluster1, img3 = cluster2)
        # save thresholded & binarised cluster map
    	cluster.to_filename(op.join(volume_dir, f'gradient-{gradient_no}_cluster-{pct[i]}-{pct[i+1]}.nii.gz'))



## STEP 2 
# investigate the independent co-activation patterns of the IFG clusters of interest

cluster_name_list={'gradient-0_cluster-0-20', 'gradient-0_cluster-80-100', 'gradient-1_cluster-0-20', 'gradient-1_cluster-80-100'}

# load neuroquery dataset
neuroquery_dset = op.join(project_dir,'data', 'neuroquery', 'neuroquery_dataset_with_abstracts.pkl.gz')
dset = Dataset.load(neuroquery_dset)

# Run independent MACM analyses for each cluster
for cluster_name in cluster_name_list:
	# load cluster
	cluster_img = nib.load(op.join(volume_dir, f'{cluster_name}.nii.gz'))
	# extract studies of interest
	roi_ids = dset.get_studies_by_mask(cluster_img) # to identify studies with at least one coordinate in the ROI.
	dset_sel = dset.slice(roi_ids) # Create a reduced version of the Dataset including only studies identified above.
	print(f"{len(roi_ids)}/{len(dset.ids)} studies report at least one coordinate in the ROI")
	# run ALE analysis
	ale = ALE(kernel__fwhm=15, null_method="approximate", n_cores=-1)
	ale.fit(dset_sel)
	corr = FWECorrector(method="montecarlo", n_iters=10000, n_cores=-1) # multiple comparisons correction
	cres = corr.transform(ale.results)
	#save results
	# create temporary output directory
	tmp_out_dir = op.join(out_dir, '{cluster_name}') # set temporary output path
	os.makedirs(tmp_out_dir, exist_ok=True) # create temporary output directory
	# save resulting maps
	cres.save_maps(tmp_out_dir, prefix=f'{cluster_name}')
	# threshold z image at p< 0.05
	z_clust = threshold_img(cres.get_map("z_level-voxel_corr-FWE_method-montecarlo"), 1.65)
	nib.save(z_clust, op.join(tmp_out_dir,f'{cluster_name}_z_level-voxel_corr-FWE_method-montecarlo_thresholded_0.05.nii.gz'))
    # visualize thresholded map
    plotting.plot_img_on_surf(z_clust, threshold = 1.65,
                              views=['lateral', 'medial'],
                              hemispheres=['left', 'right'],
                              colorbar=True, vmax = 5.5,
                              title = f'{cluster_name} map: p<0.05, voxel-level FWE corrected',
                              output_file = op.join(tmp_out_dir, f'{cluster_name}_z_level-voxel_corr-FWE_method-montecarlo_thresholded_0.05.png'))

## STEP 3
# compare the co-activation patterns of the IFG clusters

# define function to run contrast analyses 
def contrast_maps(cluster1, cluster2):
	# load clusters
	cluster_img1 = nib.load(op.join(volume_dir, f'{cluster1}.nii.gz'))
	cluster_img2 = nib.load(op.join(volume_dir, f'{cluster2}.nii.gz'))
	# Extract neuroquery studies of interest
	roi_ids1 = dset.get_studies_by_mask(cluster_img1) # to identify studies with at least one coordinate in the ROI.
	dset1 = dset.slice(roi_ids1) # Create a reduced version of the Dataset including only studies identified above.
	roi_ids2 = dset.get_studies_by_mask(cluster_img2) 
	dset2 = dset.slice(roi_ids2)
	# ALE subtraction
	ale_contrast = ALESubtraction(kernel__fwhm=15, n_iters= 10000, n_cores=-1, null_method="approximate")
	contrast_results = ale_contrast.fit(dset1, dset2)
	# create temporary output directory
	tmp_out_dir = op.join(out_dir, f'{cluster1}_vs_{cluster2}' # set temporary output path
	os.makedirs(tmp_out_dir, exist_ok=True) # create temporary output directory
	# save results
	contrast_results.save_maps(tmp_out_dir, prefix=f'{cluster1}_vs_{cluster2}') # save results 

# define function to threshold and visualize contrast results and save conjunction
def correct_contrast(cluster1, cluster2, p_val, cmap='cold_hot', out_dir):
    results_dir = op.join(out_dir, f'{cluster1}_vs_{cluster2}')
    os.makedirs(results_dir, exist_ok=True)
    map_img = nib.load(op.join(results_dir, "z_desc-group1MinusGroup2.nii.gz"))

    # threshold uncorrected p<.001
    p001_uncorrected = norm.isf(p_val)
    print(f"The p<.001 uncorrected threshold is {p001_uncorrected}")
    # two sided contrast map
    thresholded_map = threshold_img(map_img, threshold=p001_uncorrected)
    thresholded_map.to_filename(op.join(results_dir, f'{cluster1}_vs_{cluster2}_z_map_unc_0.001.nii.gz'))

    plotting.plot_img_on_surf(map_img, threshold=p001_uncorrected,
                              views=['lateral', 'medial'],
                              hemispheres=['left', 'right'],
                              colorbar=False, vmax = 10,
                              inflate = True, cmap=cmap,
                              output_file = op.join(results_dir, f'{cluster1}_vs_{cluster2}_z_map_unc_0.001_two_sided.png'))
   # one-sided contrast map cluster 1 > cluster 2
    thresholded_map = threshold_img(map_img, threshold=p001_uncorrected, two_sided=False)
    thresholded_map.to_filename(op.join(results_dir, f'{cluster1}>{cluster2}_z_map_unc_0.001.nii.gz'))  
    plotting.plot_img_on_surf(thresholded_map, threshold=p001_uncorrected,
                              views=['lateral', 'medial'],
                              hemispheres=['left', 'right'],
                              colorbar= False, vmax = 10,
                              inflate = True, cmap=cmap,
                              output_file = op.join(results_dir, f'{cluster1}>{cluster2}_z_map_unc_0.001.png'))
    # opposite contrast cluster 2 > cluster 1
    neg_map_img = math_img("-img", img=map_img)
    thresholded_map = threshold_img(neg_map_img, threshold=p001_uncorrected, two_sided=False)
    thresholded_map.to_filename(op.join(results_dir, f'{cluster2}>{cluster1}_z_map_unc_0.001.nii.gz')) 
    plotting.plot_img_on_surf(thresholded_map, threshold=p001_uncorrected,
                              views=['lateral', 'medial'],
                              hemispheres=['left', 'right'],
                              colorbar=False, vmax = 10,
                              inflate = True, cmap=cmap,
                              output_file = op.join(results_dir, f'{cluster2}>{cluster1}_z_map_unc_0.001.png'))  
    # conjunction
    mask1 = nib.load(op.join(project_dir, 'meta-analytic_gradients/gradient_explore/macms', f'{cluster1}/{cluster1}_z_level-voxel_corr-FWE_method-montecarlo_thresholded_0.05.nii.gz'))
    mask2 = nib.load(op.join(project_dir, 'meta-analytic_gradients/gradient_explore/macms', f'{cluster2}/{cluster2}_z_level-voxel_corr-FWE_method-montecarlo_thresholded_0.05.nii.gz'))

    formula = "np.where(img1 * img2 > 0, np.minimum(img1, img2), 0)"
    conj = math_img(formula, img1=mask1, img2=mask2)
    conj.to_filename(op.join(results_dir, f'{cluster1}_conj_{cluster2}_z_map_unc_0.001.nii.gz'))

    plotting.plot_img_on_surf(conj, threshold=p001_uncorrected,
                              views=['lateral', 'medial'],
                              inflate = True, cmap='cool',
                              hemispheres=['left', 'right'],
                              colorbar=False, vmax = 2,
                              output_file = op.join(results_dir, f'{cluster1}_conj_{cluster2}_z_map_unc_0.001.png'))
    # mask contrast image cluster 1 > cluster 2 with cluster 1's independent MACM map 
    map1 = nib.load(op.join(results_dir, f'{cluster1}>{cluster2}_z_map_unc_0.001.nii.gz'))
    formula = "np.where(img1 * img2 > 0, img2, 0)"
    map1_masked = math_img(formula, img1=mask1, img2=map1)
    map1_masked.to_filename(op.join(results_dir, f'{cluster1}>{cluster2}_z_map_unc_0.001_masked.nii.gz'))
    plotting.plot_img_on_surf(map1_masked, threshold=p001_uncorrected,
                              views=['lateral', 'medial'],
                              inflate = True, cmap=cmap,
                              hemispheres=['left', 'right'],
                              colorbar=False, vmax = 10,
                              output_file = op.join(results_dir, f'{cluster1}>{cluster2}_z_map_unc_0.001_masked.png'))
    # mask contrast image cluster 2 > cluster 1 with cluster 2's independent MACM map 
    map2 = nib.load(op.join(results_dir, f'{cluster2}>{cluster1}_z_map_unc_0.001.nii.gz'))
    map2_masked = math_img(formula, img1=mask2, img2=map2)
    map2_masked.to_filename(op.join(results_dir, f'{cluster2}>{cluster1}_z_map_unc_0.001_masked.nii.gz'))
    plotting.plot_img_on_surf(map2_masked, threshold=p001_uncorrected,
                              views=['lateral', 'medial'],
                              inflate = True, cmap=cmap,
                              hemispheres=['left', 'right'],
                              colorbar=False, vmax = 10,
                              output_file = op.join(results_dir, f'{cluster2}>{cluster1}_z_map_unc_0.001_masked.png'))
    # final brain figure
    img1 = math_img("np.where(img > 0, -img, 0)", img=map1_masked)
    img2 = math_img("np.where(img > 0, img, 0)", img=map2_masked)
    img = math_img("img1 + img2", img1=img1, img2=img2)
    
    figure = plotting.plot_img_on_surf(img, threshold=p001_uncorrected,
                                      views=['lateral', 'medial'],
                                      inflate = True, cmap=cmap,
                                      hemispheres=['left', 'right'],
                                      colorbar=False, vmax = 10,
                                      output_file = op.join(results_dir, f'{cluster1}&{cluster2}_final.png'))

# Run contrast & conjunction analyses

# run contrast & conjunction analyses for gradient 0
contast_maps('gradient-0_cluster-0-20', 'gradient-0_cluster-80-100')
correct_contrast('gradient-0_cluster-0-20', 'gradient-0_cluster-80-100', 0.001, out_dir)
# run contrast and conjunction analyses for gradient 1
contast_maps('gradient-1_cluster-0-20', 'gradient-1_cluster-80-100')
correct_contrast('gradient-1_cluster-0-20', 'gradient-1_cluster-80-100', 0.001, out_dir)

## STEP 4
# assess overlap with canonical networks 

# brain mask
brain = load_mni152_brain_mask()

# download & prepare Yeo parcellation maps
from nilearn.datasets import fetch_atlas_yeo_2011
atlas_yeo_2011 = fetch_atlas_yeo_2011(data_dir = op.join(project_dir, 'data/networks'))
atlas_yeo = atlas_yeo_2011['thick_7']
for i in range(1,8):
    roi = math_img(f'img == {i}', img=atlas_yeo)  
    mask = index_img(roi, 0)
	mask = math_img(f'img > 0', img=mask)
	mask = resample_to_img(mask, brain, interpolation = "nearest")
	mask.to_filename(op.join(project_dir, f'data/networks/{i}_binary.nii'))

# prepare semantic network map
sem = load_img(op.join(project_dir, 'data/networks/clustercorr_vatl.img'))
sem = math_img(f'img > 0', img=sem)
sem = resample_to_img(sem, brain, interpolation = "nearest")
sem.to_filename(op.join(project_dir, 'data/networks/clustercorr_vatl_binary.nii'))

# get list of contrast & conjunction maps
map_list = glob(op.join(project_dir, 'meta-analytic_gradients/gradient_explore/macms/*/*_z_map_unc_0.001_masked.nii.gz'))
map_conj = glob(op.join(project_dir, 'meta-analytic_gradients/gradient_explore/macms/*/*_conj_*.nii.gz'))
map_list = map_list + map_conj

# get list of network masks 
mask1 = load_img('data/networks/1_binary.nii')
volume_list = glob(op.join(project_dir, 'data/networks', '*_binary.nii'))
volume_list.append(op.join(project_dir, 'data/networks/clustercorr_vatl_binary.nii'))

	
# compute overlap

results = pd.DataFrame([])

for map in map_list:
    map_name = map.split("/")[-1].split("_z")[0]
    print(map_name)
    results_dir = map.split(f"/{map_name}_z")[0]
    print(results_dir)
    
    # load map of interest 
    map_img = load_img(op.join(results_dir, map))
    map_img = math_img(f'img > 0', img=map_img)
    map_img = resample_to_img(map_img, brain, interpolation = "nearest")
    map_results = {'cluster': [f'{map_name}' for i in range(8)], 'network': ["", "", "", "", "", "", "", ""], '%overlap': np.empty(8)}
    map_results = pd.DataFrame(map_results)


    # count number of nonzero voxels in results map
    map_img_masker = NiftiMasker(mask_strategy='whole-brain-template')
    map_img_masker = map_img_masker.fit_transform(map_img)
    unique, counts = np.unique(map_img_masker, return_counts=True)
    if len(counts)==1:
        map_img_count = 0
    else:
        map_img_count  = counts[1]
    print('Total number of active voxels in', map_name, ': ', map_img_count,'; non-active: ', counts[0])
      
    # compute overlap with canonical networks     
    for i, volume in enumerate(volume_list):
        mask_name = volume.split("/")[-1].split(".nii.gz")[0]
        mask_img = load_img(volume)
        mask_overlap = math_img(f'img1 & img2', img1=mask_img, img2 = map_img)
        mask_overlap_masker = NiftiMasker(mask_strategy='whole-brain-template')
        mask_overlap_masker = mask_overlap_masker.fit_transform(mask_overlap)
        # count number of overlapping voxels
        unique, counts = np.unique(mask_overlap_masker, return_counts=True)
        if len(counts)==1:
            mask_overlap_count = 0
        else:
            mask_overlap_count  = counts[1]
        print(f'Total number of voxels overlapping with mask {i}', map_name, ': ', mask_overlap_count)
        
        map_results.at[i, 'network'] = mask_name
        map_results.at[i, '%overlap'] = mask_overlap_count*100/map_img_count
    
    results = pd.concat([results, map_results])
results.reset_index(drop=True).to_csv(op.join(project_dir, 'meta-analytic_gradients/gradient_explore/macms/Contrasts_macms_network_overlap.csv'))


## STEP 5
# Functional decoding 

# define path
out_dir = op.join(project_dir, 'meta-analytic_gradients', 'gradient_explore', 'decoding')
os.makedirs(out_dir, exist_ok=True)

# select clusters
clusters = ['gradient-0_cluster-80-100', 'gradient-0_cluster-0-20', 'gradient-1_cluster-0-20', 'gradient-1_cluster-80-100']

# perform decoding separately for each cluster
for cluster in clusters:
    cluster_img = nib.load(op.join(volume_dir, f'{cluster}.nii.gz'))
    # Extract studies of interest
    roi_ids = dset.get_studies_by_mask(cluster_img) # to identify studies with at least one coordinate in the ROI.
    print(f"{len(roi_ids)}/{len(dset.ids)} studies report at least one coordinate in the ROI {cluster}")
    # run funcitonal decoding of ROI using BrainMap chi-square method
    decoder = BrainMapDecoder()
    decoder.fit(dset)
    decoded_df = decoder.transform(ids=roi_ids)
    #save results
    decoded_df.to_csv(op.join(out_dir, f'{cluster}.csv'))
