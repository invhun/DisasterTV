![image](https://github.com/user-attachments/assets/3e7a2348-28f1-4313-b0a3-80dcc7ab435c)

# Dataset download and preprocessing

1. Download the ‘data_GDR.zip’ file from the following link: [Dataset link](https://drive.google.com/drive/u/0/folders/1ULdBFousnyn5HFV_usdpRINbOeAyVf5t)
2. Locate the downloaded file as 'GDR/data_GDR'
3. Move to './grid_preprocess' directory
4. Excute 'video2patch_image.py'

# Run
1. Download the [CLIP B/32 model](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) and locate it in 'GDR/modules' directory
2. Execute `train_disaster.sh`
   - For more details, refer to the `train_disaster.sh` file.
