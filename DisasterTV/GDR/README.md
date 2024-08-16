# Environment Setup

```
conda create -n DisasterTV python=3.9
conda activate DisasterTV
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python boto3 requests pandas
```

# Disaster Dataset Preparing

1. Download raw video from the link: https://drive.google.com/drive/folders/1hZN-kR9t0Jw46HubYG7qf3i8GT3aaAwY?usp=drive_link
2. Makte folder and place the downloaded video in 'GDR/data/video'
3. Excute 'python video2patch_image.py'

# Run


# How to Run
1. Execute `requirements.txt`
2. Prepare disaster dataset
3. Do grid image preprocessing
4. Download the [CLIP B/32 model](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)
5. Execute `train_disaster.sh`
   - For more details, refer to the `train_disaster.sh` file.
