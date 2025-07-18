'''
G2p dataset
'''
import json
import torch

from torch.utils.data import Dataset
from utils.symbols import word2id, phoneme2id, graphemes_char2id, phonemes_char2id

class G2pdataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        data_dict = json.load(open(dataset_path, 'r'))
        self.data_pairs = list(data_dict.items()) # [('jack', "JH AE1 K"),...]
    
    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        word, phoneme_seq = self.data_pairs[idx]
        return word2id(word), phoneme2id(phoneme_seq)

def collate_fn(iter_batch): # pad & descending sort & to tensor
    N = len(iter_batch) # batch size
    word_indexes, phoneme_indexes = [list(it) for it in zip(*iter_batch)]

    # add start & end token for both word and phoneme 
    [it.insert(0, graphemes_char2id['<s>']) for it in word_indexes]
    [it.append(graphemes_char2id['</s>']) for it in word_indexes]

    # Must ！ output start and end token
    [it.insert(0, phonemes_char2id['<s>']) for it in phoneme_indexes]
    [it.append(phonemes_char2id['</s>']) for it in phoneme_indexes]

    # descending sort input seq 
    word_lengths, sort_idx = torch.sort(torch.tensor([len(it) for it in word_indexes]).long(), descending= True)
    max_word_len = word_lengths[0]

    word_padded = torch.zeros((N, max_word_len)).long() # 用zeros 刚好对应 <pad> 的 id为 0，所以pad 一般放第一个

    max_phoneme_len = max([len(it) for it in phoneme_indexes])
    phoneme_padded = torch.zeros((N, max_phoneme_len)).long()
    phoneme_length = torch.zeros((N,)).long()

    for idx, idx_s in enumerate(sort_idx.tolist()):
        word_padded[idx][:word_lengths[idx]] = torch.tensor(word_indexes[idx_s]).long()
        phoneme_padded[idx][:len(phoneme_indexes[idx_s])] = torch.tensor(phoneme_indexes[idx_s]).long()
        phoneme_length[idx] = len(phoneme_indexes[idx_s])
        
    return word_padded, word_lengths, phoneme_padded, phoneme_length


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    val_set = G2pdataset('./data/data_val.json')
    val_loader = DataLoader(val_set, batch_size= 7, collate_fn=collate_fn, shuffle=True)
    for batch in val_loader:
        word_idx, word_len, phoneme_seq_idx, phoneme_len = batch
        print('grapheme batch tensor size', word_idx.size())
        print(word_idx, word_len)
        print(phoneme_seq_idx)
        print(phoneme_len)

