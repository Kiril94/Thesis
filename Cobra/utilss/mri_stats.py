# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:34:12 2021

@author: klein
"""
from utilss.basic import DotDict
import utilss.stats as stats



def get_masks_dict(df, return_tags=True):
    
    tag_dict = {}
    tag_dict['t1'] = ['T1', 't1']
    tag_dict['mpr'] = ['mprage', 'MPRAGE'] # Mostly T1
    #print('MPRAGE is always T1w')
    #tag_dict['tfe'] = ['tfe', 'TFE'] can be acquired with or without T1/ T2
    tag_dict['spgr'] = ['SPGR', 'spgr'] #primarily T1 or PD
    #print("The smartbrain protocol occurs only for philips")
    # tag_dict['smartbrain'] = ['SmartBrain']
    tag_dict['flair'] = ['FLAIR','flair', 'Flair']
    tag_dict['t2'] = ['T2', 't2']
    #tag_dict['fse'] = ['FSE', 'fse', 'TSE', 'tse']    
    tag_dict['t2s'] = ['T2\*', 't2\*']
    #tag_dict['gre']  = ['GRE', 'gre'] # can be t2*, t1 or pd
    tag_dict['dti']= ['DTI', 'dti'] 
    tag_dict['pwi'] = ['Perfusion_Weighted']
    #print("There is one perfusion weighted image (PWI)")
    tag_dict['swi'] = ['SWI', 'swi']
    tag_dict['dwi'] = ['DWI', 'dwi']
    tag_dict['adc'] = ['ADC', 'Apparent Diffusion Coefficient']
    tag_dict['gd'] = ['dotarem', 'Dotarem', 'Gd','gd', 'GD', 'Gadolinium', 'T1\+', 't1\+']
    tag_dict['stir'] = ['STIR']
    tag_dict['tracew'] = ['TRACEW'] #
    tag_dict['asl'] = ['ASL']
    tag_dict['cest'] = ['CEST']
    tag_dict['survey'] = ['SURVEY', 'Survey', 'survey']
    tag_dict['angio'] = ['TOF', 'ToF', 'tof','angio', 'Angio', 'ANGIO', 'SWAN']
    # tags that are connected to sequences that are not useful
    tag_dict['screensave'] = ['Screen Save']
    tag_dict['autosave'] = ['3D Saved State - AutoSave']
    tag_dict['b1calib'] = ['B1_Calibration', 'calib', 'Calib', 'cal', 'Cal']
    tag_dict['loc'] = ['Loc', 'loc', 'Scout', 'LOC', 'lokal']
    tag_dict['bold'] = ['BOLD']
    tag_dict['more'] = ['vessel_scout', ]
    #print("TOF:time of flight angriography, SWAN: susceptibility-weighted angiography")
    tag_dict = DotDict(tag_dict)
    
    
    mask_dict = DotDict({key : stats.check_tags(df, tag) \
                         for key, tag in tag_dict.items()})
    #mprage is always t1 https://pubmed.ncbi.nlm.nih.gov/1535892/
    mask_dict['t1'] = stats.check_tags(df, tag_dict.t1) \
        | stats.check_tags(df, tag_dict.mpr)
    mask_dict['t1'] = stats.only_first_true(mask_dict.t1, mask_dict.gd)

    #mask_dict['t1tfe'] = mask_dict.t1 & mask_dict.tfe
    mask_dict['t1spgr'] = mask_dict.t1 & mask_dict.spgr

    mask_dict['t2_flair'] = stats.only_first_true(
        stats.check_tags(df, tag_dict.t2), mask_dict.t2s)
    mask_dict['t2_noflair'] = stats.only_first_true(mask_dict.t2_flair, mask_dict.flair)# real t2

    print("we are interested in t1, t2_noflair, flair, swi, dwi, dti, angio")
    print("combine all masks with an or and take complement")

    mask_identified = mask_dict.t1
    for mask in mask_dict.values():
        mask_identified = mask_identified | mask
    mask_dict.identified = mask_identified

    mask_dict.relevant = mask_dict.t1 | mask_dict.flair | mask_dict.t2_noflair \
        | mask_dict.t2s | mask_dict.dwi | mask_dict.swi \
            | mask_dict.angio | mask_dict.adc 

    mask_dict.none = df['SeriesDescription'].isnull()       
    mask_dict.none_nid = mask_dict.none | ~mask_dict.identified #either non or not identified
    mask_dict.other = ~mask_dict.none_nid & ~mask_dict.relevant# nont none identified and non relevant
    if return_tags:
        return mask_dict, tag_dict
    else:
        return mask_dict