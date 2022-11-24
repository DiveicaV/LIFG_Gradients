#!/usr/bin/env python
# coding: utf-8

# written by Veronica Diveica

import os
import pickle
import os.path as op
import pandas as pd
import numpy as np
import nibabel as nib
from glob import glob
from nilearn.input_data import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import concat_imgs, threshold_img, math_img, load_img, resample_to_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting
from nilearn.reporting import get_clusters_table



# set paths
project_dir = os.getcwd()
volume_dir = op.join(project_dir, 'rsfc_gradients', 'gradient_explore', 'volumes')
os.makedirs(volume_dir, exist_ok=True)
out_dir = op.join(project_dir, 'rsfc_gradients', 'gradient_explore', 'sbfc')
os.makedirs(out_dir, exist_ok=True)

# load ROI
roi_img = load_img(op.join(project_dir, 'data', 'Left_IFG_ROI.nii.gz'))

# get MNI brain template
brain = load_mni152_brain_mask()


## STEP 1
# extract clusters based on gradient values

# load gradient values matrix
with open(op.join(project_dir, 'rsfc_gradients', 'gradients.pkl'), 'rb') as fo:
    gradients = pickle.load(fo)

# extract clusters based on the first two gradients
for gradient_no in range(0,2):
	gradient_img = load_img(op.join(project_dir, 'rsfc_gradients', 'maps', f'gradient-{gradient_no}.nii.gz')) # laod gradient map
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
# investigate the independent rsfc patterns of the IFG clusters of interest 
# script based on: https://github.com/htwangtw/sbfc by H.T. Wang

cluster_name_list={'gradient-0_cluster-0-20', 'gradient-0_cluster-80-100', 'gradient-1_cluster-0-20', 'gradient-1_cluster-80-100'}

# get participant IDs
pids_df = pd.read_csv(op.join(project_dir, 'hcp1200_participants-150.tsv'), sep='\t')
pids_df = pids_df['participant_id'].tolist()
pids_df = [str(item) for item in pids_df]

for cluster in cluster_name_list:
	# set up directories
	tmp_out_dir = op.join(out_dir, f'{cluster}')
	os.makedirs(out_dir, exist_ok=True)
	first_level_dir = op.join(tmp_out_dir, 'first_level')
	os.makedirs(first_level_dir, exist_ok=True)
	# load ROI
	cluster_img = nib.load(op.join(volume_dir, f'{cluster}.nii.gz'))
	cluster_img_mask = NiftiMasker(mask_img=cluster_img)
	# first level
	for ppt in pids_df:
    	if not op.isfile(op.join(first_level_dir, f"{ppt}_{cluster}_z.nii.gz")):
			# concatenate runs
    		func_files = sorted(glob(op.join(data_dir, ppt,'rfMRI_REST*_hp2000_clean_smooth_filtered.nii.gz')))
    		func_img = concat_imgs(func_files)
    		n_scans = func_img.shape[-1]
    		tr=func_img.header.get_zooms()[-1]
    		#extract ROI mean timeseries
    		roi_ts = cluster_img_mask.fit_transform(func_img)
    		roi_ts = roi_ts.mean(axis=1)
    		roi_ts = pd.DataFrame(roi_ts, columns=[cluster])
    		# build first-level design matrix
    		frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
    		design_matrix = make_first_level_design_matrix(frametimes, add_regs=roi_ts.values, add_reg_names=roi_ts.columns.tolist(), hrf_model='spm')
    		seed_contrast = np.array([1] + [0] * (design_matrix.shape[1] - 1))
    		contrast = {cluster: seed_contrast}
    		# fit first level model
    		model = FirstLevelModel(t_r=tr, subject_label = ppt, mask_img = brain)
    		model = model.fit(run_imgs=func_img, design_matrices=design_matrix)
    		# compute contrast
    		statsmaps = model.compute_contrast(contrast[cluster], output_type='z_score')
    		image_path = op.join(first_level_dir, f"{ppt}_{cluster}_z.nii.gz")
    		statsmaps.to_filename(image_path)
	# second level
	# get input
	second_level_input  = sorted(glob(op.join(first_level_dir,'*_z.nii.gz')))
	# specify design matrix
	design_matrix = pd.DataFrame([1] * len(second_level_input), columns=['intercept'])
	# fit second level model
	second_level_model = SecondLevelModel()
	second_level_model = second_level_model.fit(second_level_input, design_matrix=design_matrix)
	statsmaps2 = second_level_model.compute_contrast(output_type='all')
	# save results
	for map in statsmaps2:
    	image_path = op.join(tmp_out_dir, f"{cluster}_{map}.nii.gz")
    	statsmaps2[map].to_filename(image_path)
    # visualize standardised map
    z_map = nib.load(op.join(tmp_out_dir, f'{cluster}_z_score.nii.gz'))
    plotting.plot_img_on_surf(z_map, threshold=0.01,
                              views=['lateral', 'medial'], inflate = True,
                              hemispheres=['left', 'right'],
                              colorbar=True, vmax = 15,
                              title ='Raw z map',
                              output_file = op.join(tmp_out_dir, f'{cluster}_z_score.png'))


# visualize corrected maps (after pTFCE - see R script)
for cluster in cluster_name_list:
	tmp_out_dir = op.join(out_dir, f'{cluster}')
	img_file = glob(op.join(tmp_out_dir, f'pTFCE-z-score-map_FWER-0.05-threshold-*.nii.gz'))[0]
    pTFCE = nib.load(img_file)
    threshold = float(img_file.split("threshold-")[-1].split(".nii.gz")[0]) # get threshold for voxel-level FWE p<.05
    pTFCE = threshold_img(pTFCE, threshold=threshold, two_sided=False)
    pTFCE.to_filename(op.join(tmp_out_dir, f'{cluster}_pTFCE_map_FWER_0.05.nii.gz'))
    plotting.plot_img_on_surf(pTFCE, threshold=threshold,
                              views=['lateral', 'medial'],
                              hemispheres=['left', 'right'],
                              inflate = True,
                              colorbar=True, vmax = 25,
                              title = f'pTFCE map: p<0.05, FWER corrected',
                              output_file = op.join(tmp_out_dir, f'{cluster}_pTFCE_map_FWER_0.05_one_sided.png'))



## STEP 3
# compare the rsfc patterns of the IFG clusters

# define function to run contrast analyses 
def contrast_maps(cluster1, cluster2, n_subjects = 150):
	tmp_out_dir = op.join(out_dir, f"{cluster1}_vs_{cluster2}")
	os.makedirs(tmp_out_dir, exist_ok=True)
	# load maps
	cluster1_input  = sorted(glob(op.join(out_dir, f'{cluster1}/first_level','*_z.nii.gz')))
	cluster2_input = sorted(glob(op.join(out_dir, f'{cluster2}/first_level','*_z.nii.gz')))
	second_level_input = cluster1_input + cluster2_input
	# design matrix
	condition_effect = np.hstack(([1] * n_subjects, [- 1] * n_subjects))
	subject_effect = np.vstack((np.eye(n_subjects), np.eye(n_subjects)))
	subjects = [f'S{i:02d}' for i in range(1, n_subjects + 1)]
	paired_design_matrix = pd.DataFrame(
		np.hstack((condition_effect[:, np.newaxis], subject_effect)),
    	columns=[f'{cluster1} vs {cluster2}'] + subjects)
	# fit second level model
	second_level_model_paired = SecondLevelModel().fit(second_level_input, design_matrix=paired_design_matrix)
	stat_maps = second_level_model_paired.compute_contrast(f'{cluster1} vs {cluster2}', output_type='all')
	# save results
	for map in stat_maps:
    	image_path = op.join(tmp_out_dir, f"{cluster1}_vs_{cluster2}_{map}.nii.gz")
    	stat_maps[map].to_filename(image_path)
    # visualize maps
    z_map = nib.load(op.join(tmp_out_dir, f'{cluster1}_vs_{cluster2}_z_score.nii.gz'))
    plotting.plot_img_on_surf(z_map, threshold=0.01, inflate = True,
                              views=['lateral', 'medial'],
                              hemispheres=['left', 'right'],
                              colorbar=True, vmax = 15,
                              title ='Raw z map',
                              output_file = op.join(tmp_out_dir, f'{cluster1}_vs_{cluster2}_z_score.png'))
    # save opposite contrast map2>map1
    neg_z_map = math_img("-img", img=z_map)
    neg_z_map.to_filename(op.join(tmp_out_dir, f'{cluster1}_vs_{cluster2}_z_score_neg.nii.gz'))


# Run contrast analyses 

# run analyses for gradient 0
contrast_maps('gradient-0_cluster-0-20', 'gradient-0_cluster-80-100')
# run contrast analysis for gradient 1
contrast_maps('gradient-1_cluster-0-20', 'gradient-1_cluster-80-100')


# Visualize corrected maps (after pTFCE - see R script)

# define function to threshold, mask and visualize corrected contrast maps
def pTFCE_contrast(cluster1, cluster2):
    tmp_out_dir = op.join(out_dir, f"{cluster1}_vs_{cluster2}")

    # conjunction
    mask1 = nib.load(op.join(project_dir, 'rsfc_gradients','gradient_explore','sbfc', f'{cluster1}', f'{cluster1}_pTFCE_map_FWER_0.05.nii.gz'))
    mask2 = nib.load(op.join(project_dir, 'rsfc_gradients','gradient_explore','sbfc', f'{cluster2}', f'{cluster2}_pTFCE_map_FWER_0.05.nii.gz'))

    formula = "np.where(img1 * img2 > 0, np.minimum(img1, img2), 0)"
    conj = math_img(formula, img1=mask1, img2=mask2)
    conj.to_filename(op.join(tmp_out_dir, f'{cluster1}_conj_{cluster2}_pTFCE_map_FWER_0.05.nii.gz'))

    plotting.plot_img_on_surf(conj, threshold=0.1,
                              views=['lateral', 'medial'],
                              inflate = True, cmap='cool',
                              hemispheres=['left', 'right'],
                              colorbar=False, vmax = 1,
                              output_file = op.join(tmp_out_dir, f'{cluster1}_conj_{cluster2}_pTFCE_map_FWER_0.05.png'))
    get_clusters_table(conj, stat_threshold = 3, cluster_threshold = 50).to_csv(op.join(tmp_out_dir, f'{cluster1}_conj_{cluster2}_pTFCE_map_FWER_0.05_table.csv'))
	
	#  cluster 1 > cluster 2
	img_file = glob(op.join(tmp_out_dir, f'pTFCE-z-score-map_FWER-0.05-threshold-*.nii.gz'))[0]
    map1 = nib.load(img_file)
    threshold = float(img_file.split("threshold-")[-1].split(".nii.gz")[0])
    map1 = threshold_img(map1, threshold=threshold, two_sided=False)
    map1.to_filename(op.join(tmp_out_dir, f'{cluster1}>{cluster2}_pTFCE_map_FWER_0.05.nii.gz'))
    plotting.plot_img_on_surf(map1, threshold=threshold,
                              views=['lateral', 'medial'],
                              inflate = True,
                              hemispheres=['left', 'right'],
                              colorbar=True, vmax = 25,
                              title = f'pTFCE map: p<0.05, FWER corrected',
                              output_file = op.join(tmp_out_dir, f'{cluster1}>{cluster2}_pTFCE_map_FWER_0.05.png'))

    formula = "np.where(img1 * img2 > 0, img2, 0)"
    map1_masked = math_img(formula, img1=mask1, img2=map1)
    map1_masked.to_filename(op.join(tmp_out_dir, f'{cluster1}>{cluster2}_pTFCE_map_FWER_0.05_masked.nii.gz'))
    plotting.plot_img_on_surf(map1_masked, threshold=threshold,
                              views=['lateral', 'medial'],
                              inflate = True,
                              hemispheres=['left', 'right'],
                              colorbar=False, vmax = 35,
                              output_file = op.join(tmp_out_dir, f'{cluster1}>{cluster2}_pTFCE_map_FWER_0.05_masked.png'))
    get_clusters_table(map1_masked, stat_threshold = 3, cluster_threshold = 50).to_csv(f'{cluster1}>{cluster2}_pTFCE_map_FWER_0.05_masked_table.csv')
	
	#  cluster 2 > cluster 1
	img_file = glob(op.join(tmp_out_dir, f'pTFCE-z-score-neg-map_FWER-0.05-threshold-*.nii.gz'))[0]
    map2 = nib.load(img_file)
    threshold = float(img_file.split("threshold-")[-1].split(".nii.gz")[0])
    map2 = threshold_img(map2, threshold=threshold, two_sided=False)
    map2.to_filename(op.join(tmp_out_dir, f'{cluster2}>{cluster1}_pTFCE_map_FWER_0.05.nii.gz'))
    plotting.plot_img_on_surf(map2, threshold=threshold,
                              views=['lateral', 'medial'],
                              inflate = True,
                              hemispheres=['left', 'right'],
                              colorbar=True, vmax = 25,
                              title = f'pTFCE map: p<0.05, FWER corrected',
                              output_file = op.join(tmp_out_dir, f'{cluster2}>{cluster1}_pTFCE_map_FWER_0.05.png'))

    map2_masked = math_img(formula, img1=mask2, img2=map2)
    map2_masked.to_filename(op.join(tmp_out_dir, f'{cluster2}>{cluster1}_pTFCE_map_FWER_0.05_masked.nii.gz'))
    plotting.plot_img_on_surf(map2_masked, threshold=threshold,
                              views=['lateral', 'medial'],
                              inflate = True,
                              hemispheres=['left', 'right'],
                              colorbar=False, vmax = 35,
                              output_file = op.join(tmp_out_dir, f'{cluster2}>{cluster1}_pTFCE_map_FWER_0.05_masked.png'))
    get_clusters_table(map2_masked, stat_threshold = 3, cluster_threshold = 50).to_csv(f'{cluster2}>{cluster1}_pTFCE_map_FWER_0.05_masked_table.csv')
	

    # final brain figure
    img1 = math_img("np.where(img > 0, img, 0)", img=map1_masked)
    img2 = math_img("np.where(img > 0, -img, 0)", img=map2_masked)
    img = math_img("img1 + img2", img1=img1, img2=img2)
    
    figure = plotting.plot_img_on_surf(img, threshold=threshold,
                                      views=['lateral', 'medial'],
                                      inflate = True,
                                      hemispheres=['left', 'right'],
                                      colorbar=False, vmax = 35,
                                      output_file = op.join(tmp_out_dir, f'{cluster2}&{cluster1}_final.png'))


# Threshold, mask and visualize corrected contrast maps
# gradient 0
pTFCE_contrast('gradient-0_cluster-0-20', 'gradient-0_cluster-80-100')
# gradient 1
pTFCE_contrast('gradient-1_cluster-0-20', 'gradient-1_cluster-80-100')


## STEP 4
# assess overlap with canonical networks 

# brain mask
brain = load_mni152_brain_mask()

# get list with all maps of interest
map_list = glob(op.join(project_dir, 'rsfc_gradients/gradient_explore/sbfc/*vs*/*_masked.nii.gz'))
map_conj = glob(op.join(project_dir, 'rsfc_gradients/gradient_explore/sbfc/*vs*/*_conj_*.nii.gz'))
map_list = map_list + map_conj

# get list of network masks 
mask1 = load_img('data/networks/1_binary.nii')
volume_list = glob(op.join(project_dir, 'data/networks', '*_binary.nii'))
volume_list.append(op.join(project_dir, 'data/networks/clustercorr_vatl_binary.nii'))

# compute overlap

results = pd.DataFrame([])

for map in map_list:
    map_name = map.split("/")[-1].split("_pTFCE")[0]
    print(map_name)
    results_dir = map.split(f"/{map_name}_pTFCE")[0]
    
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
results.reset_index(drop=True).to_csv(project_dir, 'rsfc_gradients/gradient_explore/sbfc/sbfc_contrasts_network_overlap.csv')


