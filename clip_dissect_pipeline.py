import sys
sys.path.append('CLIP-dissect') 
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import cv2
import glob
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib
import similarity
import utils
import data_utils


def dissect_pipeline(d_probe, concept_set, similarity_fn, target_name, target_layer):
    '''
    Function that fixes setting for CLIP-dissect
    d_probe: images that are being used
    concept_set: concept set that are being used to use CLIP-dissect
    similarity_fn: kind of function to be used to measure 
    '''
    clip_name = 'ViT-B/16'
    d_probe = d_probe
    concept_set = concept_set
    batch_size = 50
    device = 'cuda'
    pool_mode = 'avg'

    save_dir = 'saved_activations'
    similarity_fn = similarity_fn

    target_name = target_name
    target_layer = target_layer # last layer

    utils.save_activations(clip_name = clip_name, target_name = target_name, target_layers = [target_layer], 
                       d_probe = d_probe, concept_set = concept_set, batch_size = batch_size, 
                       device = device, pool_mode=pool_mode, save_dir = save_dir)

    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')

    pil_data = data_utils.get_data(d_probe)
    save_names = utils.get_save_names(clip_name = clip_name, target_name = target_name,
                                  target_layer = target_layer, d_probe = d_probe,
                                  concept_set = concept_set, pool_mode=pool_mode,
                                  save_dir = save_dir)
    target_save_name, clip_save_name, text_save_name = save_names

    similarities, target_feats = utils.get_similarity_from_activations(target_save_name, clip_save_name, 
                                                                  text_save_name, similarity_fn, device=device)
    # Visualize
    top_vals, top_ids = torch.topk(target_feats, k=5, dim=0)
    neurons_to_check = torch.sort(torch.max(similarities, dim=1)[0], descending=True)[1][0:20]
    font_size = 14
    font = {'size'   : font_size}

    matplotlib.rc('font', **font)
    predict_lst = []
    neuron_image_indices = {}
    neuron_image_paths = {}
    fig = plt.figure(figsize=[10, len(neurons_to_check)*2])
    subfigs = fig.subfigures(nrows=len(neurons_to_check), ncols=1)

    for j, orig_id in enumerate(neurons_to_check):
        vals, ids = torch.topk(similarities[orig_id], k=5, largest=True)
        neuron_top_ids = top_ids[:, orig_id].cpu().numpy() 

        # Get the file indices for top images, ensuring indices are within range
        valid_top_indices = [idx for idx in neuron_top_ids if idx < len(pil_data.samples)]
        invalid_top_indices = [idx for idx in neuron_top_ids if idx >= len(pil_data.samples)]
        neuron_image_indices[orig_id] = valid_top_indices
        neuron_image_paths[orig_id] = [pil_data.samples[idx][0] for idx in valid_top_indices]

        subfig = subfigs[j]
        subfig.text(0.13, 0.96, "Neuron {}:".format(int(orig_id)), size=font_size)
        subfig.text(0.27, 0.96, "CLIP-Dissect:", size=font_size)
        subfig.text(0.4, 0.96, words[int(ids[0])], size=font_size)
        predict_lst.append(words[int(ids[0])])
        axs = subfig.subplots(nrows=1, ncols=5)
        for i, top_id in enumerate(top_ids[:, orig_id]):
            im, label = pil_data[top_id]
            im = im.resize([375,375])
            axs[i].imshow(im)
            axs[i].axis('off')
    plt.show()
    
    return predict_lst, neuron_image_indices


