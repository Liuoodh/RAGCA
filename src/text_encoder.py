import pickle
from tqdm import tqdm
import numpy as np
from transformers import CLIPModel, CLIPProcessor
import re
from abc import ABC, abstractmethod


class TextEncoder:
    def __init__(self, dataset, model_name, model, id2ent, id2rel):
        self.dataset = dataset
        self.model_name = model_name
        self.model = model
        self.id2ent = id2ent
        self.id2rel = id2rel

    def get_text_vec(self):
        ent2name = self.get_info('ent', 'name')
        ent2dscp = self.get_info('ent', 'dscp')
        rel2dscp = self.get_info('rel', 'dscp')
        ent_array = []
        dscp_array = []
        rel_array = []
        names = []
        descriptions = []
        rels = []
        pbar = tqdm(total=len(self.id2ent), ncols=100, desc="start extract entity and description")
        for _, ent in self.id2ent.items():
            if ent in ent2name.keys():
                name = ent2name[ent]
            else:
                tqdm.write(ent + 'has not got name!')
                name = ent
            if ent in ent2dscp.keys():
                description = name + ' ' + ent2dscp[ent]
                description = re.sub(r'[^a-zA-Z\s]', '', description)
                # words = description.split(' ')[:40]
                # description=' '.join(words)
            else:
                tqdm.write(ent + 'has not got description!')
                description = name + ' ' + name
            names.append(name)
            descriptions.append(description)
            if len(names) >= 5:
                outputs = self.extract_feature(names)
                for arr in outputs.data.cpu().numpy():
                    ent_array.append(arr)
                names = []
            if len(descriptions) >= 5:
                outputs = self.extract_feature(descriptions)
                for arr in outputs.data.cpu().numpy():
                    dscp_array.append(arr)
                descriptions = []

                pbar.update(5)
        if len(names) > 0:
            outputs = self.extract_feature(names)
            for arr in outputs.data.cpu().numpy():
                ent_array.append(arr)
        if len(descriptions) > 0:
            outputs = self.extract_feature(descriptions)
            for arr in outputs.data.cpu().numpy():
                dscp_array.append(arr)
            pbar.update(len(descriptions))
        pbar.close()
        pbar = tqdm(total=len(self.id2rel), ncols=100, desc="start extract relation")
        for _, rel in self.id2rel.items():
            if rel in rel2dscp.keys():
                name = rel + ', Its description is that ' + rel2dscp[rel]
                name = re.sub(r'[^a-zA-Z\s]', '', name)
            else:
                tqdm.write(rel + 'has not got name!')
                name = rel + ', Its description is that ' + rel
            rels.append(name)
            if len(rels) >= 5:
                outputs = self.extract_feature(rels)
                for arr in outputs.data.cpu().numpy():
                    rel_array.append(arr)
                rels = []
                pbar.update(5)
        if len(rels) > 0:
            outputs = self.extract_feature(rels)
            for arr in outputs.data.cpu().numpy():
                rel_array.append(arr)
            pbar.update(len(rels))
        pbar.close()
        return np.array(ent_array), np.array(dscp_array), np.array(rel_array)

    @abstractmethod
    def extract_feature(self, in_put):
        pass


    def do_encode(self):
        ent, dscp, rel = self.get_text_vec()
        self.save_vec(ent, dscp, rel, self.model_name)

    def save_vec(self, ent, dscp, rel, model_name):
        output_file_ent = '../embedings/' + self.dataset + '/' + model_name + '_entity_text_feature.pickle'
        output_file_dscp = '../embedings/' + self.dataset + '/' + model_name + '_description_feature.pickle'
        output_file_rel = '../embedings/' + self.dataset + '/' + model_name + '_relation_text_feature.pickle'
        with open(output_file_ent, 'wb') as out:
            pickle.dump(ent, out)
        with open(output_file_dscp, 'wb') as out:
            pickle.dump(dscp, out)
        with open(output_file_rel, 'wb') as out:
            pickle.dump(rel, out)

    def get_id_ent_rel(self, ent_id_path, rel_id_path):
        id2ent = {}
        id2rel = {}
        f = open(ent_id_path, 'r')
        ent_lines = f.readlines()
        for line in ent_lines:
            ent, id = line.strip().split()
            id2ent[id] = ent
        f = open(rel_id_path, 'r')
        rel_lines = f.readlines()
        for line in rel_lines:
            rel, id = line.strip().split()
            id2rel[id] = rel
        return id2ent, id2rel

    def get_info(self, type, info):
        file = type + '_' + info + '.txt'
        path = '../src_data/' + self.dataset + '/' + file
        infos = {}
        for line in open(path, 'r'):
            id, text = line.strip().split('\t')
            infos[id] = text

        return infos


class Clip(TextEncoder):
    def __init__(self, dataset):
        self.model_name = 'CLIP'
        self.model = CLIPModel.from_pretrained("/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(
            "/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch32")
        ent_id_path = '../embedings/' + dataset + '/ent_id'
        rel_id_path = '../embedings/' + dataset + '/rel_id'
        self.id2ent, self.id2rel = self.get_id_ent_rel(ent_id_path, rel_id_path)
        super().__init__(dataset, self.model_name, self.model, self.id2ent, self.id2rel)

    def extract_feature(self, in_put):
        inputs = self.processor(text=in_put, return_tensors="pt", truncation=True, padding=True, max_length=77)
        outputs = self.model.get_text_features(**inputs)
        return outputs



if __name__ == "__main__":
    clip = Clip('WN9')
    clip.do_encode()
