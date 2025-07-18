from model import Transformer
import torch
from config import HP
from utils.symbols import word2id, id2phoneme
import matplotlib.pyplot as plt

import os
import numpy as np


model = Transformer()
checkpoint = torch.load(r"model_save/model_75_1500.pth",map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict']) 


while 1:
    word = input("Input: ").strip()
    wordids = word2id(word.lower())
    wordids = [HP.ENCODER_SOS_IDX] +wordids +[HP.ENCODER_EOS_IDX]
    wordids = torch.tensor(wordids).unsqueeze(0)
    phonemes, attention = model.infer(wordids)
    phonemes_list = phonemes.squeeze().cpu().numpy().tolist()
    phoneme_seq = id2phoneme(phonemes_list)
    print(phoneme_seq)
    print(attention.size())
    word_tokens = ['<s>'] + list(word.lower()) +['</s>']
    phoneme_tokens = phoneme_seq.split(" ")
    atten_map_weight = torch.sum(attention.squeeze(),dim=0)
    attn_matrix = atten_map_weight.transpose(0,1).detach().cpu().numpy()

    fig,ax = plt.subplots()
    im = ax.imshow(attn_matrix)
    ax.set_xticks(np.arange(len(phoneme_tokens)))
    ax.set_yticks(np.arange(len(word_tokens)))

    ax.set_xticklabels(phoneme_tokens)
    ax.set_yticklabels(word_tokens)

    plt.setp(ax.get_xticklabels())
    ax.set_title("word-phoneme Attention Map")
    fig.tight_layout()
    plt.show()
