from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import json
from torch.utils.data import Dataset
import torch as th
import numpy as np
from dataloaders.rawvideo_util import RawVideoExtractor
from PIL import Image, ImageSequence
import torchvision.transforms as transforms

class Disaster_Rematching_DataLoader(Dataset):
    def __init__(
            self,
            real_subset,
            synthetic_subset,
            real_data_path,
            real_features_path,
            synthetic_data_path,
            synthetic_features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            inference_sentence="",
    ):
        self.real_data_path = real_data_path
        self.real_features_path = real_features_path
        self.synthetic_data_path = synthetic_data_path
        self.synthetic_features_path = synthetic_features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.size = image_resolution
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.inference_sentence = inference_sentence
        
        self.real_subset = real_subset
        self.synthetic_subset = synthetic_subset
        #assert self.subset in ["train", "val", "test", "all"]
        video_id_path_dict = {}
        video_id_path_dict["real_all"] = os.path.join(self.real_data_path, "real_list.txt")
        video_id_path_dict["synthetic_all"] = os.path.join(self.synthetic_data_path, "synthetic_list.txt")
        caption_path_dict = {}
        caption_path_dict["real_all"] = os.path.join(self.real_data_path, "real_data.json")
        caption_path_dict["synthetic_all"] = os.path.join(self.synthetic_data_path, "synthetic_data.json")
        
        video_dict = {}

        with open(video_id_path_dict[self.synthetic_subset], 'r') as fp:
            synthetic_video_ids = [itm.strip() for itm in fp.readlines()]
            
        for root, dub_dir, video_files in os.walk(self.synthetic_features_path):
                for video_file in video_files:
                    video_id_ = ".".join(video_file.split(".")[:-1])
                    if video_id_ not in synthetic_video_ids:
                        continue
                    file_path_ = os.path.join(root, video_file)
                    video_dict[video_id_] = file_path_
                      
        with open(video_id_path_dict[self.real_subset], 'r') as fp:
            real_video_ids = [itm.strip() for itm in fp.readlines()]
            
        for root, dub_dir, video_files in os.walk(self.real_features_path):
                for video_file in video_files:
                    video_id_ = ".".join(video_file.split(".")[:-1])
                    if video_id_ not in real_video_ids:
                        continue
                    file_path_ = os.path.join(root, video_file)
                    video_dict[video_id_] = file_path_
                    
        
        self.video_dict = video_dict
        combined_video_ids = synthetic_video_ids + real_video_ids
        self.synthetic_video_ids = synthetic_video_ids
        self.real_video_ids = real_video_ids
        
        with open(caption_path_dict[self.synthetic_subset], 'rb') as f:
            synthetic_captions = json.load(f)
        self.synthetic_captions = synthetic_captions
        
        with open(caption_path_dict[self.real_subset], 'rb') as f:
            real_captions = json.load(f)

        self.sample_len = 0
        self.video_id_cap_dict = {}
        for video_id in synthetic_video_ids:
            assert video_id in synthetic_captions
            cap_txt = synthetic_captions[video_id]['captions'][0]
            self.video_id_cap_dict[len(self.video_id_cap_dict)] = (video_id, cap_txt)

        for video_id in real_video_ids:
            assert video_id in real_captions
            cap_txt = real_captions[video_id]['captions'][0]
            self.video_id_cap_dict[len(self.video_id_cap_dict)] = (video_id, cap_txt)
        
        self.video_num = len(combined_video_ids)
        print("video number: {}".format(self.video_num))
        print("For synthetic_{}, video number: {}".format(self.synthetic_subset, self.video_num))
        print("For real_{}, video number: {}".format(self.real_subset, self.video_num))

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.sample_len = len(self.video_id_cap_dict)

    def __len__(self):
        return self.sample_len

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.longlong)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float32)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.longlong)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.longlong)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.longlong)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids
    
    def __getitem__(self, idx):
        video_id, caption = self.video_id_cap_dict[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        # print(choice_video_ids)
 
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, choice_video_ids


