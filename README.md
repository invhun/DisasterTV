# DisasterTV

The official code implementation of DisasterTV. we introduce a text-video pair dataset and an efficient retrieval model for disaster research. 

# Environment Setup
```
conda create -n DisasterTV python=3.9
conda activate DisasterTV
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

#from CLIP
pip install opencv-python boto3 requests pandas
pip install ftfy regex tqdm fvcore
```

# Dataset Construction (Matching Filtering Module)

Please refer to the 'DisasterTV/matching_filtering' directory to construct our disaster dataset.

![image](https://github.com/user-attachments/assets/71f85d59-c899-48bf-ab1d-4afd07089859)

# Text-Video Retrieval (GDR)

Please refer to the 'DisasterTV/GDR' directory to run an our efficient retrieval model, Grid-based Disaster Retrieval (GDR).

![image](https://github.com/user-attachments/assets/3e7a2348-28f1-4313-b0a3-80dcc7ab435c)


# Acknowledgments

This repository is based on [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)
