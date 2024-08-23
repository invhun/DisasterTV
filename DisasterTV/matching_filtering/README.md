![image](https://github.com/user-attachments/assets/71f85d59-c899-48bf-ab1d-4afd07089859)


# Disaster Dataset Preprocessing

1. Download raw video from the link: https://drive.google.com/drive/folders/1hZN-kR9t0Jw46HubYG7qf3i8GT3aaAwY?usp=drive_link
2. Make directory and put the downloaded video in 'GDR/data/video'
3. Move to './grid_preprocess' directory
4. Excute 'video2patch_image.py'

# Run
1. Download the [CLIP B/32 model](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) and put it in 'GDR/modules' directory
2. Execute `train_disaster.sh`
   - For more details, refer to the `train_disaster.sh` file.
