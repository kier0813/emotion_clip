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
from collections import Counter

def get_actual_labels(image_indices, pil_data):
    labels = []
    for idx in image_indices:
        path, _ = pil_data.samples[idx]
        label = path.split('/')[-2]  
        labels.append(label)
    return labels

def neuron_accuracy(neuron_image_indices, predicted_emotions, pil_data):
    detailed_results = {}

    for neuron, predicted_emotion in zip(neuron_image_indices.keys(), predicted_emotions):
        neuron_id = neuron.item() 
        image_indices = neuron_image_indices[neuron]
        actual_labels = get_actual_labels(image_indices, pil_data)
        
        detailed_results[neuron_id] = {
            'predicted_emotion': predicted_emotion,
            'actual_labels': actual_labels,
            'top_image_indices': image_indices,
        }

        correct_count = actual_labels.count(predicted_emotion)
        
        # Calculate the accuracy for this neuron
        accuracy = correct_count / len(image_indices)
        detailed_results[neuron_id]['accuracy'] = accuracy

    # Calculate overall accuracy
    overall_accuracy = sum(d['accuracy'] for d in detailed_results.values()) / len(detailed_results)

    # Print the detailed results
    for neuron_id, results in detailed_results.items():
        print(f"Neuron {neuron_id}: Predicted Emotion = {results['predicted_emotion']}, Accuracy = {results['accuracy']}")
        for image_idx, actual_label in zip(results['top_image_indices'], results['actual_labels']):
            print(f"    Image Index {image_idx}: Actual Label = {actual_label}")
    print(f"Overall accuracy: {overall_accuracy}")