import os
import random
from PIL import Image
import numpy as np

import torch
import torch.utils
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, UnidentifiedImageError

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def get_transform(img_size):
    # mid_sz=math.floor(img_size*1.25)
    # mid_reso=(mid_sz,mid_sz)
    transform_list=[
        transforms.Resize((img_size,img_size), interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        # transforms.RandomCrop((img_size,img_size)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    return transforms.Compose(transform_list)


# class CSVDataset(torch.utils.data.Dataset):
#     def __init__(self, arrowset, classifier_free_training_prob=0.1, resolution=256, cogvlm=False):
#         super().__init__()
#         self.arrowset = arrowset
#         self.classifier_free_training_prob = classifier_free_training_prob
#         self.resolution = resolution
#         self.negative_tag=""
#         self.transform=get_transform(img_size=resolution)
#         self.cogvlm=cogvlm
    
#     def __len__(self):
#         return self.arrowset.num_rows
    
#     def __getitem__(self, index):
#         return self.get_item_from_arrowset(index)
    
#     def get_item_from_arrowset(self, index):
#         #id,blobstore_key,caption,width,height,clarity_score,aesthetic_score,nr_quality_avg_score_nn,category,image_save_path
#         image_path = self.arrowset[index]['image_save_path']
#         image = Image.open(image_path)
#         if not image.mode == "RGB":
#             image = image.convert("RGB")
#         img_w, img_h = image.size
#         img_size=min(img_h,img_w)
#         crop_upx=(img_w-img_size)//2;crop_upy=(img_h-img_size)//2
#         crop_downx=crop_upx+img_size;crop_downy=crop_upy+img_size
#         image = image.crop((crop_upx,crop_upy,crop_downx,crop_downy))
#         image = self.transform(image)

#         if self.cogvlm:
#             prompt = self.arrowset[index]['cogvlm_caption']
#         else:
#             prompt = self.arrowset[index]['caption']
#         if prompt is None:
#             prompt = "empty caption"
#             print(f"Empty Caption of Image {image_path}")
#         if self.classifier_free_training_prob > 0.0 and len(prompt)>20 and random.random() < self.classifier_free_training_prob:
#             prompt = self.negative_tag
        
#         return {"image": image, "prompt": prompt}


# class CSVDatasetInfer(torch.utils.data.Dataset):
#     def __init__(self, arrowset, classifier_free_training_prob=0.1, resolution=256, cogvlm=False):
#         super().__init__()
#         self.arrowset = arrowset
#         self.classifier_free_training_prob = classifier_free_training_prob
#         self.resolution = resolution
#         self.negative_tag=""
#         self.transform=get_transform(img_size=resolution)
#         self.cogvlm=cogvlm
    
#     def __len__(self):
#         return self.arrowset.num_rows
    
#     def __getitem__(self, index):
#         return self.get_item_from_arrowset(index)
    
#     def get_item_from_arrowset(self, index):
#         #id,blobstore_key,caption,width,height,clarity_score,aesthetic_score,nr_quality_avg_score_nn,category,image_save_path
#         image_path = self.arrowset[index]['image_save_path']
#         image = Image.open(image_path)
#         if not image.mode == "RGB":
#             image = image.convert("RGB")
#         img_w, img_h = image.size
#         img_size=min(img_h,img_w)
#         crop_upx=(img_w-img_size)//2;crop_upy=(img_h-img_size)//2
#         crop_downx=crop_upx+img_size;crop_downy=crop_upy+img_size
#         image = image.crop((crop_upx,crop_upy,crop_downx,crop_downy))
#         image = self.transform(image)

#         if self.cogvlm:
#             prompt = self.arrowset[index]['cogvlm_caption']
#         else:
#             prompt = self.arrowset[index]['caption']
#         if prompt is None:
#             prompt = "empty caption"
#             print(f"Empty Caption of Image {image_path}")
#         if self.classifier_free_training_prob > 0.0 and len(prompt)>20 and random.random() < self.classifier_free_training_prob:
#             prompt = self.negative_tag
        
#         return {"image": image, "prompt": prompt, "id": self.arrowset[index]['id']}

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, arrowset, classifier_free_training_prob=0.1, resolution=256, cogvlm=False):
        super().__init__()
        self.arrowset = arrowset
        self.classifier_free_training_prob = classifier_free_training_prob
        self.resolution = resolution
        self.negative_tag = ""
        self.transform = get_transform(img_size=resolution)
        self.cogvlm = cogvlm

    def __len__(self):
        return self.arrowset.num_rows

    def __getitem__(self, index):
        try:
            return self.get_item_from_arrowset(index)
        except (OSError, UnidentifiedImageError, ValueError) as e:
            # --- 跳过损坏图像 ---
            bad_path = self.arrowset[index]['image_save_path']
            print(f"[Warning] Skip broken image at index {index}: {bad_path} ({e})")

            # 递归地尝试下一个样本（保证batch不会少）
            new_index = (index + 1) % len(self.arrowset)
            return self.__getitem__(new_index)

    def get_item_from_arrowset(self, index):
        image_path = self.arrowset[index]['image_save_path']

        # --- 加载图片并处理 ---
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_w, img_h = image.size
        img_size = min(img_h, img_w)
        crop_upx = (img_w - img_size) // 2
        crop_upy = (img_h - img_size) // 2
        crop_downx = crop_upx + img_size
        crop_downy = crop_upy + img_size
        image = image.crop((crop_upx, crop_upy, crop_downx, crop_downy))
        image = self.transform(image)

        # --- 加载文本 ---
        prompt = self.arrowset[index]['cogvlm_caption'] if self.cogvlm else self.arrowset[index]['caption']
        if not prompt:
            prompt = "empty caption"
            print(f"[Info] Empty caption for {image_path}")

        if self.classifier_free_training_prob > 0.0 and len(prompt) > 20 and random.random() < self.classifier_free_training_prob:
            prompt = self.negative_tag

        return {"image": image, "prompt": prompt}


class CSVDatasetInfer(torch.utils.data.Dataset):
    def __init__(self, arrowset, classifier_free_training_prob=0.1, resolution=256, cogvlm=False):
        super().__init__()
        self.arrowset = arrowset
        self.classifier_free_training_prob = classifier_free_training_prob
        self.resolution = resolution
        self.negative_tag = ""
        self.transform = get_transform(img_size=resolution)
        self.cogvlm = cogvlm

    def __len__(self):
        return self.arrowset.num_rows

    def __getitem__(self, index):
        try:
            return self.get_item_from_arrowset(index)
        except (OSError, UnidentifiedImageError, ValueError) as e:
            bad_path = self.arrowset[index]['image_save_path']
            print(f"[Warning] Skip broken image at index {index}: {bad_path} ({e})")
            new_index = (index + 1) % len(self.arrowset)
            return self.__getitem__(new_index)

    def get_item_from_arrowset(self, index):
        image_path = self.arrowset[index]['image_save_path']
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_w, img_h = image.size
        img_size = min(img_h, img_w)
        crop_upx = (img_w - img_size) // 2
        crop_upy = (img_h - img_size) // 2
        crop_downx = crop_upx + img_size
        crop_downy = crop_upy + img_size
        image = image.crop((crop_upx, crop_upy, crop_downx, crop_downy))
        image = self.transform(image)

        prompt = self.arrowset[index]['cogvlm_caption'] if self.cogvlm else self.arrowset[index]['caption']
        if not prompt:
            prompt = "empty caption"
            print(f"[Info] Empty caption for {image_path}")

        if self.classifier_free_training_prob > 0.0 and len(prompt) > 20 and random.random() < self.classifier_free_training_prob:
            prompt = self.negative_tag

        return {"image": image, "prompt": prompt, "id": self.arrowset[index]['id']}
    
def extract_path(path, key_str='laion-5b'):
    start_index = path.lower().find(key_str.lower())
    if start_index == -1:
        raise ValueError(f"The key_str '{key_str}' was not found in the path '{path}'.")
    sub_path = path[start_index:]
    return os.path.splitext(sub_path)[0]

class PrecomputedCSVDataset(torch.utils.data.Dataset):
    def __init__(self, arrowset, precomputed_rootdir):
        super().__init__()
        self.arrowset = arrowset
        self.precomputed_rootdir = precomputed_rootdir
    
    def __len__(self):
        return self.arrowset.num_rows
    
    def __getitem__(self, index):
        return self.get_item_from_arrowset(index)
    
    def get_item_from_arrowset(self, index):
        #id,blobstore_key,caption,width,height,clarity_score,aesthetic_score,nr_quality_avg_score_nn,category,image_save_path
        image_path = self.arrowset[index]['image_save_path']
        clean_path = extract_path(image_path)
        text_feature_path = os.path.join(self.precomputed_rootdir, 'clip_feature', clean_path+'.npz')
        vae_feature_path = os.path.join(self.precomputed_rootdir, 'vae_feature', clean_path+'.npy')

        vae_feature = torch.from_numpy(np.load(vae_feature_path))
        text_feature = np.load(text_feature_path)
        encoder_hidden_states = torch.from_numpy(text_feature['encoder_hidden_states'])
        attn_mask = torch.from_numpy(text_feature['attn_mask'])
        pooled_embed = torch.from_numpy(text_feature['pooled_embed'])  
        
        return {"image": vae_feature, "prompt_embeds": (encoder_hidden_states, attn_mask, pooled_embed)}

class PrecomputedCSVDatasetInfer(torch.utils.data.Dataset):
    def __init__(self, arrowset, precomputed_rootdir):
        super().__init__()
        self.arrowset = arrowset
        self.precomputed_rootdir = precomputed_rootdir
    
    def __len__(self):
        return self.arrowset.num_rows
    
    def __getitem__(self, index):
        return self.get_item_from_arrowset(index)
    
    def get_item_from_arrowset(self, index):
        #id,blobstore_key,caption,width,height,clarity_score,aesthetic_score,nr_quality_avg_score_nn,category,image_save_path
        image_path = self.arrowset[index]['image_save_path']
        clean_path = extract_path(image_path)
        text_feature_path = os.path.join(self.precomputed_rootdir, 'clip_feature', clean_path+'.npz')
        vae_feature_path = os.path.join(self.precomputed_rootdir, 'vae_feature', clean_path+'.npy')

        vae_feature = torch.from_numpy(np.load(vae_feature_path))
        text_feature = np.load(text_feature_path)
        encoder_hidden_states = torch.from_numpy(text_feature['encoder_hidden_states'])
        attn_mask = torch.from_numpy(text_feature['attn_mask'])
        pooled_embed = torch.from_numpy(text_feature['pooled_embed'])  
        
        return {"image": vae_feature, "prompt_embeds": (encoder_hidden_states, attn_mask, pooled_embed), "id": self.arrowset[index]['id']}