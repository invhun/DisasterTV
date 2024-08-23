import os
import glob
from multiprocessing import Pool
import time
from video_preprocessing import video_to_patch_image

def process_video(video_file):
    base_name = os.path.basename(video_file)
    name, ext = os.path.splitext(base_name)
    print("processing:", name)
    output_file = os.path.join(output_image_path, name + ".jpg")
    video_to_patch_image(video_file, output_file, 16)

if __name__ == '__main__':
    input_video_path = "../data_GDR/video/"
    output_image_path = "../data_GDR/grid_image/image_16patch/"
    video_files = glob.glob('../data_GDR/video/*.[mg][pi][f4]')
    os.makedirs(output_image_path)

    with Pool(processes=8) as pool: 
        pool.map(process_video, video_files)


