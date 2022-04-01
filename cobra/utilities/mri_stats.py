# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:34:12 2021

@author: klein
"""
from utilities.basic import DotDict
import utilities.stats as stats
import pandas as pd

def get_masks_dict(df, return_tags=True):

    tag_dict = {}
    tag_dict['t1'] = ['T1', 't1', 'BRAVO', ]
    tag_dict['mpr'] = ['mprage', 'MPRAGE']  # Mostly T1
    #print('MPRAGE is always T1w')
    # tag_dict['tfe'] = ['tfe', 'TFE'] can be acquired with or without T1/ T2
    tag_dict['spgr'] = ['SPGR', 'spgr']  # primarily T1 or PD
    #print("The smartbrain protocol occurs only for philips")
    # tag_dict['smartbrain'] = ['SmartBrain']
    tag_dict['flair'] = ['FLAIR', 'flair', 'Flair']
    tag_dict['t2'] = ['T2', 't2']
    #tag_dict['fse'] = ['FSE', 'fse', 'TSE', 'tse']
    tag_dict['t2s'] = ['T2\*', 't2\*']
    tag_dict['gre']  = ['GRE', 'gre', 'FGRE'] # can be t2*, t1 or pd
    tag_dict['dti'] = ['DTI', 'dti']
    tag_dict['pwi'] = ['Perfusion_Weighted']
    #print("There is one perfusion weighted image (PWI)")
    tag_dict['swi'] = ['SWI', 'swi', 'SUSCEPTABILITET']
    tag_dict['dwi'] = ['DWI', 'dwi', 'MUSE', 'Diffusion']
    tag_dict['adc'] = ['ADC', 'Apparent Diffusion Coefficient', 'adc']
    tag_dict['gd'] = ['dotarem', 'Dotarem', 'Gd', 'gd', 
                    'GD', 'Gadolinium', 'T1\+', 't1\+']
    tag_dict['stir'] = ['STIR', 'stir']
    tag_dict['tracew'] = ['TRACEW', 'travew']
    tag_dict['asl'] = ['ASL', 'asl']
    tag_dict['cest'] = ['CEST']
    tag_dict['survey'] = ['SURVEY', 'Survey', 'survey']
    tag_dict['angio'] = ['TOF', 'ToF', 'tof', 'angio', 'Angio', 'ANGIO', 'SWAN',
                         'PCA', 'pca', 'dce', 'PC', 'pc', 'TRANCE', 'trance',
                         'mIP', 'MIP', ]
    tag_dict['pd'] = ['PDW']
    # tags that are connected to sequences that are not useful
    tag_dict['screensave'] = ['Screen Save']
    tag_dict['autosave'] = ['3D Saved State - AutoSave']
    tag_dict['b1calib'] = ['B1_Calibration', 'calib', 'Calib', 'cal', 'Cal',
                           'CAL', ]
    tag_dict['loc'] = ['Loc', 'loc', 'Scout', 'LOC', 'lokal', 'LOKAL']
    tag_dict['bold'] = ['BOLD']
    tag_dict['mip'] = ['MINIP', 'MInimum Intensity', 
                       'Min Intensity Projection', 'Min Intensity ',
                       'Minimum Intensity Projection','Tra SWIp MinIP']
    tag_dict['more'] = ['vessel_scout', 'VRT', 'csf_flow', 'WIP',
                        'svs', 'SVS', 'animation', 'ACOM', 'Verification',
                        'BA', 'BLACK_BLOOD', 'Batch', 'Cerebral Blood Flow',
                        'DISPLAY', 'IMPAX Volume', 'MYELO',
                        'MI Reading', 'MM Oncology Reading', 'SCREENSAVE',
                        'SPINE',
                        ]
    #print("TOF:time of flight angriography, SWAN: susceptibility-weighted angiography")
    tag_dict = DotDict(tag_dict)

    mask_dict = DotDict({key: stats.check_tags(df, tag)
                         for key, tag in tag_dict.items()})
    # mprage is always t1 https://pubmed.ncbi.nlm.nih.gov/1535892/
    mask_dict['t1'] = stats.check_tags(df, tag_dict.t1) \
        | stats.check_tags(df, tag_dict.mpr)
    mask_dict['t1gd'] = mask_dict.t1 & mask_dict.gd
    mask_dict['t1'] = stats.only_first_true(mask_dict.t1, mask_dict.gd)

    #mask_dict['t1tfe'] = mask_dict.t1 & mask_dict.tfe
    mask_dict['t1spgr'] = mask_dict.t1 & mask_dict.spgr

    mask_dict['t2'] = stats.only_first_true(
        mask_dict.t2, mask_dict.flair)  # no flair
    mask_dict['t2'] = stats.only_first_true(
        mask_dict.t2, mask_dict.t2s)  # no t2s
    mask_dict['t2gd'] = mask_dict.t2 & mask_dict.gd
    mask_dict['gd'] = stats.only_first_true(
        mask_dict.gd, mask_dict.t2gd)  # no t2
    mask_dict['gd'] = stats.only_first_true(
        mask_dict.gd, mask_dict.t1gd)  # no t1
    
    mask_identified = mask_dict.t1
    for mask in mask_dict.values():
        mask_identified = mask_identified | mask
    mask_dict.identified = mask_identified

    mask_dict.relevant = mask_dict.t1 | mask_dict.flair | mask_dict.t2 \
        | mask_dict.dwi | mask_dict.swi \

    mask_dict.none = df['SeriesDescription'].isnull()
    # either non or not identified
    mask_dict.none_nid = mask_dict.none | ~mask_dict.identified
    # nont none identified and non relevant
    mask_dict.other = ~mask_dict.none_nid & ~mask_dict.relevant
    if return_tags:
        return mask_dict, tag_dict
    else:
        return mask_dict
    

def get_studies_df(df_sorted, threshold=2):
    """Returns:
        Data frame containing patientid, DateTimeStart of study (first scan),
        and number of volumes
    """
    df_studies = pd.DataFrame(columns=['PatientID', 'StudyNum',
                    'DateTimeStart', 'NumVolumes'])
    
    patient_ids = df_sorted['PatientID'].unique()
    for patient in patient_ids:
        print('|', end='')
        study_num = 0
        patient_mask = df_sorted['PatientID'] == patient
        date_times = df_sorted[patient_mask]['DateTime']
        date_time0 = date_times[0]
        df_studies = df_studies.append({'PatientID':patient, 
                                        'StudyNum':study_num,
                                            'DateTimeStart':date_time0}, 
                                        ignore_index=True)
        num_volumes = 1
        for date_time in date_times[1:]:
            time_diff = date_time-date_time0
            if time_diff.total_seconds()/3600 > threshold:
                study_num+=1
                df_studies.iloc[-1, -1] = num_volumes
                df_studies = df_studies.append({'PatientID':patient, 
                                        'StudyNum':study_num,
                                        'DateTimeStart':date_time}, 
                                        ignore_index=True)
                date_time0 = date_time
                num_volumes = 1
            else:
                num_volumes +=1
        df_studies.iloc[-1, -1] = num_volumes
    return df_studies
