# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:54:15 2019

@author: WT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from main_bible import preprocessW2V, ie_preprocess

def contextise(documents_token, context_window=4):
    for sent_token in documents_token:
        pass

class NGramEmbedModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramEmbedModeler, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size*embedding_dim, 256)
        self.fc2 = nn.Linear(256, vocab_size)
    
    def forward(self, inputs):
        x = self.embedding(inputs).view((1, -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)