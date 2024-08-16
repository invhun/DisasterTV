import cv2
from PIL import Image
import numpy as np
import math
    
def video_to_patch_image(video_path, output_image_path, patch_num, patch_size=(32, 32), grid_size=(7, 7), frame_selection_method='uniform'):
    cap = cv2.VideoCapture(video_path)
    final_image_size = (patch_size[0] * grid_size[0], patch_size[1] * grid_size[1])
    final_image = np.zeros((final_image_size[0], final_image_size[1], 3), dtype=np.uint8)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    total_patches_needed = grid_size[0] * grid_size[1]
    
    if len(frames) * patch_num < total_patches_needed:
        print("Error: Video does not contain enough frames to fill the grid with the specified number of patches.")
        return
    else:
        if frame_selection_method == 'uniform':
            patches_per_frame = int(math.sqrt(patch_num))
            total_frames_needed = total_patches_needed // patch_num + (1 if total_patches_needed % patch_num != 0 else 0)
            step = max(len(frames) // total_frames_needed, 1)
            selected_frames = [frames[i] for i in range(0, len(frames), step)][:total_frames_needed]

    count = 0
    for frame in selected_frames:
        resized_frame = cv2.resize(frame, (patch_size[0] * patches_per_frame, patch_size[1] * patches_per_frame), interpolation=cv2.INTER_AREA)
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        for i in range(patches_per_frame):
            for j in range(patches_per_frame):
                if count < total_patches_needed:
                    patch = resized_frame[i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]]
                    row = count // grid_size[1]
                    col = count % grid_size[1]
                    final_image[row*patch_size[0]:(row+1)*patch_size[0], col*patch_size[1]:(col+1)*patch_size[1], :] = patch
                    count += 1
                else:
                    break

    cap.release()
    
    final_pil_image = Image.fromarray(final_image)
    final_pil_image.save(output_image_path)


# Example usage
# video_path = "/video.mp4"
# output_image_path = 'output_image.jpg'
# video_to_patch_image(video_path, output_image_path, 16)

