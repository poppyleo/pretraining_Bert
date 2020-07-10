import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import numpy as np


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        d_ff = d_model * 4
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def attention(query, key, value, mask=None, dropout=None):
    """
        Compute 'Scaled Dot Product Attention'
        query, key, value : batch_size, n_head, seq_len, dim of space
        :return output[0]: attention output. (batch_size, n_head, seq_len, head_dim)
        :return output[1]: attention score. (batch_size, n_head, seq_len, seq_len)
    """

    d_k = query.size(-1)
    # scores: batch_size, n_head, seq_len, seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    # auto-regression
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MHPooling(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        "Take in model size and number of heads."
        super(MHPooling, self).__init__()
        assert d_model % h == 0
        # d_model = num_heads * head_dim
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # auto-regressive
        attn_shape = (1, 3000, 3000)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1).cuda()

    def forward(self, x):
        """Implements Figure 2
        :param x: (batch_size, max_len, input_dim=d_model=output_dim)
        :return:
        """

        nbatches, seq_len, d_model = x.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x, x, x))]  # (batch_size, num_heads, max_len, head_dim)

        # 2) Apply attention on all the projected vectors in batch.
        # output(batch_size, n_head, seq_len, head_dim), attention score
        x, self.attn = attention(query, key, value, mask=self.mask[:, :, :seq_len, :seq_len],
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize):
        super(LocalRNN, self).__init__()
        """
        LocalRNN structure
        input_dim = output_dim
        """
        self.ksize = ksize
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_dim, output_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, output_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(input_dim, output_dim, batch_first=True)

        # self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())

        # To speed up
        # [0,...,k_size-1, 1,...,k_size, 2,...,k_size+1 ... len-k_size,...,len-1]
        idx = [i for j in range(self.ksize - 1, 10000, 1) for i in range(j - (self.ksize - 1), j + 1, 1)]
        self.select_index = torch.LongTensor(idx).cuda()
        self.zeros = torch.zeros((self.ksize - 1, input_dim)).cuda()  # (ksize-1, input_dim)

    def forward(self, x):
        """
        :param x: (batch_size, max_len, input_dim)
        :return: h: (batch_size, max_len, input_dim=d_model=output_dim)
        """
        nbatches, l, input_dim = x.shape
        x = self.get_K(x)  # b x seq_len x ksize x d_model
        batch, l, ksize, d_model = x.shape
        # input: (batch_size*max_len, ksize, d_model)
        # output: (batch_size*max_len, 1, d_model)
        h = self.rnn(x.view(-1, self.ksize, d_model))[0][:, -1, :]
        return h.view(batch, l, d_model)

    def get_K(self, x):
        """将输入加滑动窗口
        :param x: (batch_size, max_len, input_dim)
        :return: key: split to kernel size. (batch_size, l, ksize, input_dim)
        """
        batch_size, l, d_model = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, ksize-1, input_dim=d_model)
        x = torch.cat((zeros, x), dim=1)  # (batch_size, max_len+ksize-1, input_dim)
        key = torch.index_select(x, 1, self.select_index[:self.ksize * l])  # (batch_size, ksize*l, input_dim)
        key = key.reshape(batch_size, l, self.ksize, -1)  # (batch_size, l, ksize, input_dim)
        return key


class LocalRNNLayer(nn.Module):
    "Encoder is made up of attconv and feed forward (defined below)"

    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize)
        self.connection = SublayerConnection(output_dim, dropout)

    def forward(self, x):
        "Follow Figure 1 (left) for connections. Res connection."
        x = self.connection(x, self.local_rnn)
        return x


class Block(nn.Module):
    """
    One Block.
    a transformer encoder
    """

    def __init__(self, input_dim, output_dim, rnn_type, ksize, N, h, dropout):
        super(Block, self).__init__()
        # get model list
        # rnn + res net + layer norm
        self.layers = clones(
            LocalRNNLayer(input_dim, output_dim, rnn_type, ksize, dropout), N)
        self.connections = clones(SublayerConnection(output_dim, dropout), 2)
        self.pooling = MHPooling(input_dim, h, dropout)
        self.feed_forward = PositionwiseFeedForward(input_dim, dropout)

    def forward(self, x):
        n, l, d = x.shape
        # local rnn
        for i, layer in enumerate(self.layers):
            x = layer(x)
        # multi-head attention
        x = self.connections[0](x, self.pooling)
        # ffn(non-liner)
        x = self.connections[1](x, self.feed_forward)
        return x


class RTransformer(nn.Module):
    """
    The overall model
    """

    def __init__(self,config,rnn_type ='gru',ksize = 1 , n=16):
        """
        :param d_model: config.num_attention_heads*int(config.hidden_size / config.num_attention_heads)
        :param rnn_type: 'rnn','lstm','gru'
        :param ksize: kernel_size
        :param config.num_hidden_layers: num of encoders
        :param n: num of local-rnn layers
        :param h: num of heads
        :param dropout: dropout prop
        """
        super(RTransformer, self).__init__()
        N = n
        self.d_model = config.num_attention_heads*int(config.hidden_size / config.num_attention_heads)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.norm = LayerNorm(self.d_model)
        self.feed_forward = PositionwiseFeedForward(self.d_model,config.attention_probs_dropout_prob)

        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(
                Block(self.d_model, self.d_model, rnn_type, ksize, N=N, h=config.num_attention_heads, dropout=self.dropout))
        self.forward_net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.forward_net(x)
        return x
