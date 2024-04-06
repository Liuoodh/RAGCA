import torch.nn
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
import imagehash
import pickle
from pytorch_pretrained_vit import ViT
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, CLIPModel

class ImageEncoder():

    TARGET_IMG_SIZE = 224
    img_to_tensor = transforms.ToTensor()
    Normalizer = transforms.Normalize((0.5,), (0.5,))
    def __init__(self,dataset):
        self.dataset=dataset
    @staticmethod
    def get_embedding(self, filter_gate='PHash'):
        pass

    # 给定一个实体的所有图片 过滤出一张图
    def filter(self, ent_imgs_dir,filter_gate):
        image_names = os.listdir(ent_imgs_dir)
        if len(image_names)==0:
            print(ent_imgs_dir + "is empty.")
            return Image.new("RGB", (384, 384), (0, 0, 0))
        if filter_gate=='PHash':
            hash_dict = {}
            for filename in image_names:
                image_path = os.path.join(ent_imgs_dir, filename)
                image = Image.open(image_path)
                phash = imagehash.phash(image)
                hash_dict[filename] = phash
            target_image = None
            min_distance = float('inf')
            for target_image_filename, target_image_hash in hash_dict.items():
                distance = 0
                for filename, phash in hash_dict.items():
                    if filename != target_image_filename:
                        distance += target_image_hash - phash  # 汉明距离
                if distance < min_distance:
                    min_distance = distance
                    target_image = target_image_filename
            return Image.open(os.path.join(ent_imgs_dir, target_image)).convert('RGB').resize((224, 224))

    # 特征提取
    def extract_feature(self, base_path,filter_gate):
        print("start extract")
        self.model.eval()
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        dict = {}

        ents = os.listdir(base_path)

        pbar = tqdm(total=len(ents))

        while len(ents) > 0:
            # print(len(ents))
            ents_50 = []
            ents_50_ok = []

            for i in range(5):
                if len(ents) > 0:
                    ent = ents.pop()
                    try:
                        ents_50.append(ent)
                        # ents_50.append(base_path + ent + '/' + os.listdir(base_path + ent + '/')[0])
                    except Exception as e:
                        print(e)
                        continue


            tensors=[]
            for ent in ents_50:

                img = self.filter(os.path.join(base_path,ent),filter_gate=filter_gate)

                img_tensor = self.img_to_tensor(img)
                img_tensor = self.Normalizer(img_tensor)
                if img_tensor.size()[0] == 3:
                    tensors.append(img_tensor)
                    if self.dataset=='FB15K-237':
                        ents_50_ok.append(ent.replace('-','/')) # 替换为实体id
                    if self.dataset=='WN18':
                        ents_50_ok.append(ent[1:]) # 去掉n
                    if self.dataset== 'FB15K':
                        ents_50_ok.append('/'+ent.replace('.','/'))
                else:
                    print(ent)
                    print(img_tensor.shape)
            if len(tensors) > 0:
                tensor = torch.stack(tensors, 0)
                tensor = tensor.cuda()

            result = self.model(Variable(tensor))
            result_npy = result.data.cpu().numpy()
            for i in range(len(result_npy)):
                dict[ents_50_ok[i]] = result_npy[i]
            pbar.update(5)
        pbar.close()
        return dict


class VisionTransformer(ImageEncoder):
    def __init__(self,dataset):
        super(VisionTransformer, self).__init__(dataset)
        self.model = ViT('B_16_imagenet1k', pretrained=True)

    def get_embedding(self, base_path,filter_gate='PHash'):
        self.model.eval()
        self.model.cuda()
        self.d = self.extract_feature(base_path,filter_gate=filter_gate)
        return self.d

    def save_embedding(self, output_file):
        with open(output_file, 'wb') as out:
            pickle.dump(self.d, out)

class Clip(ImageEncoder):
    def __init__(self,dataset):
        super(Clip, self).__init__(dataset)
        self.model = CLIPModel.from_pretrained("/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch32")
    def get_embedding(self, base_path,filter_gate='PHash'):
        self.d = self.extract_feature(base_path,filter_gate=filter_gate)
        return self.d
    # 节点、节点图片、节点描述、关系+描述 四类特征
    def extract_feature(self, base_path,filter_gate= 'PHash'):
        print("start extract entity!")
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        dict = {}

        ents = os.listdir(base_path)

        pbar = tqdm(total=len(ents))

        while len(ents) > 0:
            # print(len(ents))
            ents_50 = []
            ents_50_ok = []

            for i in range(5):
                if len(ents) > 0:
                    ent = ents.pop()
                    try:
                        ents_50.append(ent)
                        # ents_50.append(base_path + ent + '/' + os.listdir(base_path + ent + '/')[0])
                    except Exception as e:
                        print(e)
                        continue
            imgs=[]
            for ent in ents_50:
                img = self.filter(os.path.join(base_path,ent),filter_gate=filter_gate)
                imgs.append(img)
                if self.dataset=='FB15K-237':
                    ents_50_ok.append(ent.replace('-','/')) # 替换为实体id
                if self.dataset=='WN18' or self.dataset=='WN9':
                    ents_50_ok.append(ent[1:]) # 去掉n
                if self.dataset== 'FB15K':
                    ents_50_ok.append('/'+ent.replace('.','/'))

            if len(imgs) > 0:
                inputs = self.processor(images=imgs, return_tensors="pt")

            result = self.model.get_image_features(**inputs)

            result_npy = result.data.cpu().numpy()
            for i in range(len(result_npy)):
                dict[ents_50_ok[i]] = result_npy[i]
            pbar.update(5)
        pbar.close()
        return dict

if __name__ == "__main__":

    model = Clip(dataset='WN9')
    base_path = '/data/dataset/knowledge_graph/WN18_IMG/images/'
    img_vec = model.get_embedding(base_path, filter_gate='PHash')
    f = open('../embedings/WN9/ent_id', 'r')
    Lines = f.readlines()

    id2ent = {}
    img_array = []
    dim = 512
    count=0
    for l in Lines:
        ent, id = l.strip().split()
        id2ent[id] = ent
        if ent in img_vec.keys():
            print(id, ent)
            img_array.append(img_vec[ent])
        else:
            count+=1
            img_array.append(np.zeros(shape=(dim,)))

    output_file = '../embedings/WN9/CLIP_img_feature.pickle'
    img_array = np.array(img_array)
    with open(output_file, 'wb') as out:
        pickle.dump(img_array, out)
    print(count)