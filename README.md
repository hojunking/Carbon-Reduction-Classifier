## Carbon-Reduction Activity Image Classification Project
 **Industry-Academic Project (Capstone Design 1 lecture of KNU) for 1st semester of 2023**  
 
 **Team Members: Hojun Song, Jeongwon Cha, Yunhee Koo, Sehyun Park**


### Quick Start
- Environment setting  
    ```
    git clone https://github.com/hojunking/Carbon-Reduction-Classifier.git
    pip install -r requirements.txt
    ```
- Download model weight (checkpoints)
    https://drive.google.com/drive/folders/1PkaVoGZ88wTsj0pFubkDy50zdm2NOR1M?usp=sharing  
    Put in *./inception_resnet_v2/*  

- Run with a carbon reduction image
    ```
    python carbon_classifier.py --image_path img_path
    ```


