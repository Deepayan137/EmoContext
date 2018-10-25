import os
import torch
import numpy as np
import pdb


def convert_toTensor(pairs):
    pairs = np.array(pairs)
    return torch.from_numpy(pairs)

def gmkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split(samples, **kwargs):
    total = len(samples)
    indices = list(range(total))
    if kwargs['random']:
        np.random.shuffle(indices) 
    percent = kwargs['split']
    # Split indices
    current = 0
    train_count = np.int(percent*total)
    train_indices = indices[current:current+train_count]
    current += train_count
    test_indices = indices[current:]
    # pdb.set_trace()
    train_subset, test_subset = [], []
    for i in train_indices:
        try:
            train_subset.append(samples[i])
        except Exception as e:
            pass
    for i in test_indices:
        try:
            test_subset.append(samples[i])
        except Exception as e:
            pass
    # train_subset = [samples[i] for i in train_indices]
    # test_subset = [samples[i] for i in test_indices]
    return train_subset, test_subset

def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")



# def pad_seq(data):

#     def max_len():
#         lengths = [data.data[i]['length'] for i in range(len(data))]
#         return max(lengths)
#     # pdb.set_trace()
#     max_length = max_len()
#     for idx in data.data:
#         length = data.data[idx]['length']
#         difference = max_length - length
#         vector_size = data.data[idx]['feature'].shape[1]
#         padding = np.zeros((difference, vector_size), dtype=np.float32)
#         data.data[idx]['feature'] = np.concatenate((data.data[idx]['feature'], padding),axis=0)
#     return data

from sklearn.metrics import f1_score
class Eval:
    def __init__(self, lmap):

        self.lmap = lmap
        self.ilmap=  {v:k for k,v in lmap.items()}

    def decode(self, prediction):
        # feats_dist = prediction.data.view(-1).div(self.T).exp()
        _, top_i = torch.max(prediction.data, 1)
        return top_i

    def predict(self, prediction, target):
        prediction = self.decode(prediction)
        if prediction == target:
            return 1.0
        return 0.0
    def f1(self, prediction, target):
        prediction = self.decode(prediction)
        prediction = prediction.cpu().numpy()
        target = target.cpu().numpy()
        return f1_score(target, prediction, average='weighted')
class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total = 0
        self.max = -1*float("inf")
        self.min = float("inf")

    def add(self, element):
        # pdb.set_trace()
        self.total += element
        self.count += 1
        self.max = max(self.max, element)
        self.min = min(self.min, element)

    def compute(self):
        # pdb.set_trace()
        if self.count == 0:
            return float("inf")
        return self.total/self.count

    def __str__(self):
        return "%s (min, avg, max): (%.3lf, %.3lf, %.3lf)"%(self.name, self.min, self.compute(), self.max)
