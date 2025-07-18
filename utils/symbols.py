'''
 字形graphemes和音素phonemes互相转换
'''
# jack -> <s> j a c k </s>
graphemes = ["<pad>", "<s>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")  # 29
# <s> JH AE1 K </s>
phonemes = ["<pad>", "<s>", "</s>"] + [
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 
    'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 
    'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 
    'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 
    'Z', 'ZH'
]   # 72

graphemes_id2char = dict(enumerate(graphemes)) # {0:'<pad>',1:'<s>',...,...}
phonemes_id2char = dict(enumerate(phonemes))  # {0:'<pad>',1:'<s>',...,...}
graphemes_char2id = dict((v,k) for k,v in graphemes_id2char.items())
phonemes_char2id = dict((v,k) for k, v in phonemes_id2char.items())


def word2id(word):  # 'jack' -> [1,13，24,...,2]
    return [graphemes_char2id[c] for c in list(word)]

def id2word(idx_list): # [1,13，24,...,2] ->  'jack' 
    return ''.join([graphemes_id2char[c] for c in idx_list])

def phoneme2id(phoneme_seq): # 'JH AE1 K' -> [2， 35， 28]
    return [phonemes_char2id[p] for p in phoneme_seq.split(" ")]

def id2phoneme(idx_list):
    return ' '.join([phonemes_id2char[idx] for idx in idx_list])

if __name__ == "__main__":
    word = 'jack'
    phoneme = 'JH AE1 K'
    print(word2id(word))
    print(phoneme2id(phoneme))
    print(id2word(word2id(word)))
    print(id2phoneme(phoneme2id(phoneme)))

