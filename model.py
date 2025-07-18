'''
transformer 
'''
import math
import torch

import torch.nn.functional as F
import torch.nn as nn
from config import HP

class PositionalEncoding(nn.Module):
    '''
    PositionalEncoding 不带参数更新
    PositionalEmbedding 带有参数更新
    '''
    def __init__(self, d_model, max_len = 10000):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0))/d_model)

        pe[0,:,0::2] = torch.sin(position * div_term)
        pe[0,:,1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:,0:x.size(1):,:]

        return x


class Encoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.token_embedding = nn.Embedding(HP.grapheme_size, HP.encoder_dim)
        self.pe = PositionalEncoding(HP.encoder_dim, HP.encoder_max_input)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(HP.encoder_layer)])
        self.drop = nn.Dropout(HP.encoder_drop_prob)
        self.register_buffer('scale', torch.sqrt(torch.tensor(HP.encoder_dim).float()))

    def forward(self, inputs, input_mask):
        token_emb = self.token_embedding(inputs)
        inputs = self.pe(token_emb*self.scale) 

        for idx, layer in enumerate(self.layers):
            inputs = layer(inputs, input_mask)
        return inputs


class EncoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.self_att_layer_norm = nn.LayerNorm(HP.encoder_dim)
        self.pff_layer_norm = nn.LayerNorm(HP.encoder_dim)

        self.self_attn = MultiHeadAttentionLayer(HP.encoder_dim, HP.nhead)
        self.pff = PointWiseFeedForwardLayer(HP.encoder_dim,HP.encoder_feed_forward_dim, HP.ffn_drop_prob)
        self.drop = nn.Dropout(HP.encoder_drop_prob)

    def forward(self, inputs, inputs_mask):
        _inputs, att_res = self.self_attn(inputs, inputs, inputs, inputs_mask)
        inputs = self.self_att_layer_norm(inputs+self.drop(_inputs))
        _inputs = self.pff(inputs)
        inputs = self.pff_layer_norm(inputs+self.drop(_inputs))

        return inputs

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, nhead):
        super().__init__()
        self.hid_dim = hid_dim
        self.nhead = nhead
        assert not self.hid_dim % self.nhead
        self.head_dim = self.hid_dim // self.nhead

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.register_buffer('scale', torch.sqrt(torch.tensor(hid_dim).float()))
    
    def forward(self, query, key, value, inputs_mask = None):
        N = query.size(0)

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(N, -1, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(N, -1, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(N, -1, self.nhead, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if inputs_mask is not None:
            energy = torch.masked_fill(energy, inputs_mask == 0 , -1.e10)
        attention = F.softmax(energy, dim = -1)

        out = torch.matmul(attention, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(N, -1, self.hid_dim)
        out = self.fc_o(out)

        return out, attention

class PointWiseFeedForwardLayer(nn.Module):
    def __init__(self, hid_dim, pff_dim, pff_drop_out):
        super().__init__()

        self.hid_dim = hid_dim
        self.pff_dim = pff_dim
        self.pff_drop_out = pff_drop_out

        self.fc1 = nn.Linear(self.hid_dim, self.pff_dim)
        self.fc2 = nn.Linear(self.pff_dim, self.hid_dim)

        self.dropout = nn.Dropout(self.pff_drop_out)
    
    def forward(self, inputs): 
        inputs = self.dropout(F.relu(self.fc1(inputs)))
        out = self.fc2(inputs)

        return out

class Decoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.token_embedding = nn.Embedding(HP.phoneme_size, HP.decoder_dim)
        self.pe = PositionalEncoding(HP.decoder_dim, HP.MAX_DECODE_STEP)

        self.layers = nn.ModuleList([Decoder_layer() for _ in range(HP.decoder_layer)])
        self.fc_out = nn.Linear(HP.decoder_dim, HP.phoneme_size)
        self.drop = nn.Dropout(HP.decoder_drop_prob)
        self.register_buffer('scale', torch.sqrt(torch.tensor(HP.decoder_dim).float()))

    def forward(self, trg, enc_src, trg_mask, src_mask):
        token_emb = self.token_embedding(trg)
        pos_emb = self.pe(token_emb*self.scale)
        trg = self.drop(pos_emb)

        for idx,layer in enumerate(self.layers):
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        out = self.fc_out(trg)
        return out,attention

class Decoder_layer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_self_att = MultiHeadAttentionLayer(HP.decoder_dim, HP.nhead)
        self.mask_self_norm = nn.LayerNorm(HP.decoder_dim)

        self.mha = MultiHeadAttentionLayer(HP.decoder_dim, HP.nhead)
        self.mha_norm = nn.LayerNorm(HP.decoder_dim)

        self.pff = PointWiseFeedForwardLayer(HP.decoder_dim, HP.decoder_feed_forward_dim, HP.ffn_drop_prob)
        self.pff_norm = nn.LayerNorm(HP.decoder_dim)

        self.dropout = nn.Dropout(HP.decoder_drop_prob)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.mask_self_att(trg, trg, trg, trg_mask)
        trg = self.mask_self_norm(trg + self.dropout(_trg))

        _trg, attention = self.mha(trg, enc_src, enc_src, src_mask)
        trg = self.mha_norm(trg + self.dropout(_trg))

        _trg = self.pff(trg)
        trg = self.pff_norm(trg + self.dropout(_trg))

        return trg, attention
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @staticmethod
    def create_src_mask(src):
        mask = (src != HP.ENCODER_PAD_IDX).unsqueeze(1).unsqueeze(2).to(HP.device)
        return mask
    
    @staticmethod
    def cretea_trg_mask(trg):
        trg_len = trg.size(1)
        pad_mask = (trg != HP.DECODER_PAD_IDX).unsqueeze(1).unsqueeze(2).to(HP.device)
        sub_mask = torch.tril(torch.ones((trg_len, trg_len),dtype = torch.uint8)).bool()

        trg_mask = pad_mask & sub_mask

        return trg_mask


    def forward(self, src, trg):
        src_mask = self.create_src_mask(src)
        trg_mask = self.cretea_trg_mask(trg)

        # print("src mask",src_mask)
        # print("trg mask", trg_mask)
        enc_src = self.encoder(src, src_mask)
    
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        print(output.shape)
        return output, attention
    def infer(self, x): # "Jack"  -> "JH AE1 K"
        batch_size = x.size(0)
        src_mask = self.create_src_mask(x)
        enc_src = self.encoder(x, src_mask)

        trg = torch.zeros(size=(batch_size, 1)).fill_(HP.DECODER_SOS_IDX).long().to(HP.device)

        decoder_step = 0
        while True:
            if decoder_step == HP.MAX_DECODE_STEP:
                print("Warning: Reached Max Decoder step")
                break
            trg_mask = self.cretea_trg_mask(trg)
            output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(-1)[:,-1]
            trg = torch.cat((trg, pred_token.unsqueeze(0)),dim=-1)
            if pred_token == HP.DECODER_EOS_IDX:
                print("decoder done")
                break
            decoder_step+=1
        return trg[:,1:],attention



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from datasets import G2pdataset,collate_fn
    
    net = Transformer()
    test_set = G2pdataset('./data/data_test.json')
    test_loader = DataLoader(test_set, batch_size=3, collate_fn= collate_fn)

    for batch in test_loader:
        words_idx ,words_len, phoneme_seqs_idx, phoneme_len = batch
        print(words_idx.size())
        print(phoneme_seqs_idx.size())

        out, att = net(words_idx, phoneme_seqs_idx)

        break