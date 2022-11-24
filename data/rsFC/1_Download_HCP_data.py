#!/usr/bin/env python
# coding: utf-8

import os
import os.path as op
import pandas as pd
import boto3
import botocore

# provide your access key to the hcp aws repository
ACCESS_KEY_id = ''
ACCESS_KEY_password = ''

# define function to download hcp files of interest
def download_ppt(hcp_data_dir=None, pid=None):
    boto3.Session(aws_access_key_id=ACCESS_KEY_id, aws_secret_access_key=ACCESS_KEY_password)
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hcp-openaccess')

    for tmp_pid in pid:
        print(tmp_pid)
        s3_keys = bucket.objects.filter(Prefix='HCP_1200/{}/MNINonLinear/Results/rfMRI'.format(str(tmp_pid)))
        s3_keylist = [key.key for key in s3_keys]

        rsfmri_clean = [i for i in s3_keylist if i.endswith(('_hp2000_clean.nii.gz'))]
        rsfmri_clean = [i for i in rsfmri_clean if '7T' not in i]

        for tmp_run in rsfmri_clean:
            run_dir = tmp_run.split('/')[-2]
            os.makedirs(op.join(hcp_data_dir, str(tmp_pid), run_dir), exist_ok=True)

            rsfmri_download_file = op.join(hcp_data_dir, str(tmp_pid), run_dir, op.basename(tmp_run))
            with open(rsfmri_download_file, 'wb') as f:
                bucket.download_file(tmp_run, rsfmri_download_file)

            for mask in ['CSF', 'WM']:
                tmp_run_mask = tmp_run.replace('_hp2000_clean.nii.gz', '_{}.txt'.format(mask))
                mask_download_file = op.join(hcp_data_dir, str(tmp_pid), run_dir, op.basename(tmp_run_mask))
                with open(mask_download_file, 'wb') as f:
                    bucket.download_file(tmp_run_mask, mask_download_file)


# specify directory
hcp_dir = os.getcwd()
hcp_data_dir = op.join(hcp_dir, 'hcp-openaccess', 'HCP1200')
# load list of participants to download
pids_df = pd.read_csv(op.join(hcp_dir, 'hcp1200_participants-150.tsv'), sep='\t')

# download data
download_ppt(hcp_data_dir=hcp_data_dir, pid=pids_df['participant_id'].tolist())
