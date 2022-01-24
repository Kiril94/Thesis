
import os

command = "python preprocess.py preprocess-yolo -if '/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/rCMB_DefiniteSubject' -of /home/neus/Documents/09.UCPH/MasterThesis/DATA/prova_training_YOLO -lf /home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/tables/SynthCMB/rCMBInformationInfo.csv -ts 0.20 -vs 0.10 --overwrite"
command = "python preprocess.py preprocess-yolo '/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/NoCMBSubject' '/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/rCMB_DefiniteSubject' 'Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/sCMB_DefiniteSubject' '/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/sCMB_NoCMBSubject' -of '/media/neus/USB DISK/prova_training_YOLO' -lf /home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/tables/SynthCMB/all_info.csv -ts 0.20 -vs 0.10 --overwrite"

os.system(f'cd process_data_CMBdetection; {command}') 