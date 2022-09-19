"""
    Save new patient IDs to text file (cases excluded) which contain keywords 
    related to microbleeds.
"""
#%%
import PyPDF2
import glob
import pandas as pd
from utilities.basic import get_part_of_path
#%%
with open("C:\\Users\\kiril\\OneDrive - University of Copenhagen\\Cobra\\CMB_paper\\radiologistCMBwords.txt", 'r') as f:
    words = [line.strip() for line in f]
print('search for', words)

df_cases = pd.read_csv("G:\\CoBra\\Data\\swi_nii\\cmb_study\\cases_v5.csv")
cases_pids = [case_pid[:-7] for case_pid in df_cases.new_name]
pats_with_cmbs = []
i = 0
for file in glob.glob("G:\\CoBra\\Data\\swi_nii\\cmb_study\\reports\\*\\*\\*.pdf"):
    pat_id = get_part_of_path(file,6,7)
    if pat_id in cases_pids:
        print('skip case')
        continue
    #if i>200:
     #   break
    i+=1
    with open(file, 'rb') as f:
        pdfReader = PyPDF2.PdfFileReader(f)
        cmb_present=False
        for page in range(pdfReader.numPages):
            pageObject = pdfReader.getPage(page)
            txt = pageObject.extractText()
            str_list = txt.split()
            word_num = 0
            while (not cmb_present) and (word_num<len(words)):
                word = words[word_num]
                if (word in str_list):
                    index = str_list.index(word)
                    if str_list[index-1] in ['ikke', 'ingen']:
                        print(pat_id, 'ikke or ingen')
                        pass
                    else:
                        cmb_present = True
                word_num += 1
    if cmb_present:
        pats_with_cmbs.append(pat_id)
        print(pat_id, 'has cmb')
print(i, 'files analysed')
print(len(pats_with_cmbs), 'patients with cmbs')
with open("G:\\CoBra\\Data\\swi_nii\\cmb_study\\reports_with_cmb_cases_excluded.txt", mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(pats_with_cmbs))