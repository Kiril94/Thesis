import pandas as pd
from scipy.stats import ks_2samp


def get_dV_df(brain_regions_ls, sids, case_control_dic, pred_df):
    """Compute rate of brain volume change for brain_regions_ls in cm^3/year.
    For SeriesInstanceUIDs in sids and patients in case_control_dic."""
    # list of all cases and controls
    all_patients = [item for sublist in list(case_control_dic.values()) for item in sublist]\
        + list(case_control_dic.keys())
    # select patients and sids of interest
    pred_df = pred_df[pred_df.PatientID.isin(all_patients)] 
    pred_df = pred_df[pred_df.SeriesInstanceUID.isin(sids)]
    
    # Compute date differences between scans and store in extra df
    pred_df['Date_diff'] = pred_df.groupby('PatientID')['InstanceCreationDate'].apply(lambda x: x.dt.date - x.min().date())# compute difference in dates
    df_pat = pd.DataFrame(pred_df.groupby('PatientID').Date_diff.max()).reset_index()
    # create patient groups and sort by date diff, Date_diff==0 is first scan Date_diff>0 is second scan
    pred_df2 = pred_df.groupby(['PatientID']).apply(lambda x: (x.sort_values('Date_diff', ascending=False)))
    # select brain regions of interest and compute difference in volume
    df_differences = pred_df2[brain_regions_ls].pct_change(-1)                
    # select first entry (1st scan V - 2nd scan V)                
    df_differences = df_differences.reset_index(drop=False).drop_duplicates('PatientID', keep='first').drop(columns=['level_1'])
    df_differences = pd.merge(df_pat, df_differences, on='PatientID', how='outer')
    assert len(df_differences)==len(all_patients), 'length of dataframe is not equal to the number of all patients'
    # turn timedelta to days
    df_differences['Date_diff'] = df_differences.Date_diff.map(lambda x: x.days)
    # compute change rate of brain volume in %/year
    df_differences.iloc[:, 2:] = df_differences[brain_regions_ls].div(df_differences.Date_diff, axis=0)*100 #absolute change per day in percent
    return df_differences

def compute_unadj_pvals(df_dV, case_control_dic, brain_regions_ls):
    controls = [item for sublist in list(case_control_dic.values()) for item in sublist]
    cases =  list(case_control_dic.keys())
    pval_dic = {}
    df_dV_pos = df_dV[df_dV.PatientID.isin(cases)]
    df_dV_neg = df_dV[df_dV.PatientID.isin(controls)]
    for brain_reg in brain_regions_ls:
        stats, pval = ks_2samp(df_dV_pos[brain_reg].to_numpy(), df_dV_neg[brain_reg].to_numpy())
        pval_dic[brain_reg] = pval
    return pval_dic