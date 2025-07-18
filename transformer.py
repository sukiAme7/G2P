'''
Transformer的另一种写法
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    '''
    Transformer 位置编码实现
    Args:
        d_model:每一个Embedding后词向量的长度
        max_seq_len: 输入序列的最大长度
    '''
    
    def __init__(self,d_model: int, max_seq_len: int):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1) # (max_seq_len,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0))/d_model
        )  # 10000^(-2i/d)  == exp(-2i*log(10000)/d) 这种写法更加稳定
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # 会自动完成self.pe = pe，同时将pe登记为buffer不可学习的参数

    def forward(self, x: torch.Tensor):
        n, seq_len, d_model = x.shape
        pe: torch.tensor = self.pe
        assert seq_len <= pe.shape[1]
        assert d_model == pe.shape[2]

        return x*d_model**0.5 + pe[:, 0:seq_len, : ]
        
def attention(query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              mask = None):
    '''
    注意力计算  Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
    '''
    # q shape : [N, heads, q_len, d_k]
    # k shape : [N, heads, k_len, d_k]
    # v shape : [N, heads, k_len, d_v]
    assert query.shape[-1] == key.shape[-1]
    d_k = key.shape[-1]
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = scores.softmax(dim=-1)

    return torch.matmul(scores, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int , d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0

        # assume d_k = d_v
        self.d_k = d_model // heads #假如 512//8 = 64 是将输入分成8个头 而并非复制8份
        self.heads = heads
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model)  # W_Q 将Embedding 映射为 Q
        self.k = nn.Linear(d_model, d_model)  # W_K 将Embedding 映射为 K
        self.v = nn.Linear(d_model, d_model)  # W_V 将Embedding 映射为 V
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):
        n, q_len = query.shape[0:2]
        n, k_len = key.shape[0:2]

        # 观察这个操作 可以发现 输入 Q K V是被分成了 比如8个头  而不是复制成了8份
        q_ = self.q(query).reshape(n, q_len, self.heads, self.d_k).transpose(1,2) # [N, heads, seq_len, d_k]

        k_ = self.k(key).reshape(n, k_len, self.heads, self.d_k).transpose(1,2)
        v_ = self.v(value).reshape(n, k_len, self.heads, self.d_k).transpose(1,2)
        # print(q_.shape,k_.shape)
        attention_res = attention(q_, k_, v_, mask)
        concat_res = attention_res.transpose(1, 2).reshape(
            n, q_len, self.d_model) # [N, seq_len, d_model] 最终输出 与输入 embedding shape一致
        concat_res = self.dropout(concat_res)

        return self.out(concat_res)


class FeedForward(nn.Module):
    '''
    前向网络
    '''
    def __init__(self, d_mdoel: int ,d_ff: int ,dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_mdoel, d_ff)
        self.layer2 = nn.Linear(d_ff, d_mdoel)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.dropout(F.relu(out1))
        out3 = self.layer2(out2)

        return out3
    
class EncoderLayer(nn.Module):
    def __init__(self, heads: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p = dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p = dropout)

    def forward(self, x, src_mask = None):
        tmp = self.self_attention(x, x, x, src_mask) #自注意力的体现 Q K V 皆为自身
        tmp = self.dropout1(tmp)
        out1 = self.norm1(x + tmp) # add 残差操作
        tmp = self.ffn(out1)
        tmp = self.dropout2(tmp)
        out2 = self.norm2(out1 + tmp)
        return out2

class DecoderLayer(nn.Module):
    def __init__(self, heads: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.cross_attention = MultiHeadAttention(heads, d_model, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_kv, dst_mask, src_dst_mask):
        tmp = self.self_attention(x, x, x, dst_mask)
        tmp = self.dropout1(tmp)
        out1 = self.norm1(x + tmp)

        tmp = self.cross_attention(out1, encoder_kv, encoder_kv, src_dst_mask)
        tmp = self.dropout2(tmp)
        out2 = self.norm2(out1 + tmp)

        tmp = self.ffn(out2)
        tmp = self.dropout3(tmp)
        out3 = self.norm3(out2 + tmp)

        return out3
        
class Encoder(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,  # 论文中的 Nx
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int =120):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.pe = PositionalEncoding(d_model, max_seq_len)

        self.layers = []
        for i in range(n_layers):
            self.layers.append(EncoderLayer(heads, d_model, d_ff, dropout))
        self.layers = nn.ModuleList(self.layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask = None):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,  # 论文中的 Nx
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int =120):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.pe = PositionalEncoding(d_model, max_seq_len)

        self.layers = []
        for i in range(n_layers):
            self.layers.append(DecoderLayer(heads, d_model, d_ff, dropout))
        self.layers = nn.ModuleList(self.layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_kv, dst_mask = None, src_dst_mask = None):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_kv, dst_mask, src_dst_mask)
        return x

class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size: int,
                 dst_vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 200):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, pad_idx, d_model, d_ff,
                               n_layers, heads, dropout, max_seq_len)
        self.decoder = Decoder(dst_vocab_size, pad_idx, d_model, d_ff,
                               n_layers, heads, dropout, max_seq_len)
        self.pad_idx = pad_idx
        self.output_layer = nn.Linear(d_model, dst_vocab_size)

    def generate_mask(self,
                      q_pad: torch.Tensor,
                      k_pad: torch.Tensor,
                      with_left_mask: bool = False):
        # q_pad shape: [n, q_len]
        # k_pad shape: [n, k_len]
        # q_pad k_pad dtype: bool
        assert q_pad.device == k_pad.device
        n, q_len = q_pad.shape
        n, k_len = k_pad.shape

        mask_shape = (n, 1, q_len, k_len)
        if with_left_mask:
            mask = 1 - torch.tril(torch.ones(mask_shape))
        else:
            mask = torch.zeros(mask_shape)
        mask = mask.to(q_pad.device)
        for i in range(n):
            mask[i, :, q_pad[i], :] = 1
            mask[i, :, :, k_pad[i]] = 1
        mask = mask.to(torch.bool)
        return mask

    def forward(self, x, y):

        src_pad_mask = x == self.pad_idx
        dst_pad_mask = y == self.pad_idx
        src_mask = self.generate_mask(src_pad_mask, src_pad_mask, False)
        dst_mask = self.generate_mask(dst_pad_mask, dst_pad_mask, True)
        src_dst_mask = self.generate_mask(dst_pad_mask, src_pad_mask, False)
        encoder_kv = self.encoder(x, src_mask)
        res = self.decoder(y, encoder_kv, dst_mask, src_dst_mask)
        res = self.output_layer(res)
        return res
