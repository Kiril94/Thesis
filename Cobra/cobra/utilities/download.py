import os
from os.path import join
from datetime import datetime
import shutil
import pandas as pd
import glob


def move_files_from_sif(df_group, df_volume_dir, df_patient_dir,
                        dst_dir, 
                        patient_log_file, volume_log_file,
                        download_docs=True, src_dir="Y:\\" ):
    """Moving patients in patient_list, starting from index stored in patient_log_file
    to dst_dir 
    df_group: volume ids for every patient, 
    volume_dir_dic: dictionary with directories (e.g. 2019_01/02ac123...) for every volume
    patient_dir_dic: dictionary with directories (e.g. 2019_01/02ac123...) for every patient"""
    
    volume_dir_dic = pd.Series(
    df_volume_dir.Directory.values, index=df_volume_dir.SeriesInstanceUID)\
        .to_dict()
    patient_dir_dic = pd.Series(
        df_patient_dir.Directory.values, index=df_patient_dir.PatientID)\
            .to_dict()

    try:
        with open(volume_log_file) as f:
            volume_lines = f.readlines()
        last_volume_path = volume_lines[0]
        print(f"Try to remove {last_volume_path}")
    except Exception as e:
        print("ERROR : "+str(e))
    
    try:
        shutil.rmtree(last_volume_path)
    except Exception as e:
        print("ERROR : "+str(e))

    try:
        print("Get index of the last patient")
        with open(patient_log_file) as f:
            lines = f.readlines()
        last_patient_idx = int(lines[-2][6:11]) 
    except Exception as e:
        last_patient_idx = 0
        print("ERROR : "+str(e))
        print("Start with first patient in the list.")

    patient_list = df_group.PatientID.unique()
    counter = last_patient_idx
    for pat in patient_list[last_patient_idx:]:
        counter += 1
        patient_dir = patient_dir_dic[pat]
        print(f"Patient: {patient_dir}", end='\n')
        print(datetime.now())
        log_str = f"{patient_dir}\nindex: {counter}\
                \n {datetime.now()}\n"
        with open(patient_log_file, mode="a+") as f:
            f.write(log_str)
        # Copy doc files
        if download_docs:
            print("Download reports")
            if not os.path.exists(join(dst_dir, patient_dir, 'DOC')):
                os.makedirs(join(dst_dir, patient_dir, 'DOC'))
            for doc_path_src in glob.iglob(join(src_dir, patient_dir,"*","DOC","*","*",".pdf")):
                doc_path_src = os.path.normpath(doc_path_src)
                study_id = doc_path_src.split(os.sep)[3]
                doc_id = doc_path_src.split(os.sep)[5]
                dst_doc_dir = join(dst_dir, patient_dir, 'DOC')
                doc_path_dst = join(dst_doc_dir, f"{study_id}_{doc_id}.pdf")
                try:
                    shutil.copy(doc_path_src, doc_path_dst)
                except Exception as e:
                    print("ERROR : "+str(e))
                
        # copy dcm files
        volumes = df_group[df_group.PatientID==pat]['SeriesInstanceUID']
        print(f"download {len(volumes)} volumes")
        for volume in volumes:
            try:
                volume_dir = volume_dir_dic[volume]
            except:
                print("volume not in dict")
                continue
            try:
                volume_src = os.path.normpath(join(src_dir, volume_dir))
            except Exception as e:
                print("ERROR : "+str(e))
                continue
            if len(os.listdir(volume_src))==0:
                print('-',  end='')
                continue
            else:        
                volume_uid = volume_src.split(os.sep)[-1]
                volume_dst = join(dst_dir, patient_dir, volume_uid)
                try:
                    with open(volume_log_file, mode="w") as f:
                        f.write(volume_dst)
                    shutil.copytree(volume_src, volume_dst)
                    print("|",  end='')
                except Exception as e:
                    print("ERROR : "+str(e))
