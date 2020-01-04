import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models


class rnn_encoder(nn.Module):

    def __init__(self, config, embedding=None):
        super(rnn_encoder, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.embedding = embedding if embedding is not None else nn.Embedding(config.src_vocab_size, config.emb_size)
        if config.cell == 'gru':
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                              num_layers=config.enc_num_layers, dropout=config.dropout,
                              bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=config.enc_num_layers, dropout=config.dropout,
                               bidirectional=config.bidirectional)

    def forward(self, inputs, lengths):
        embs = pack(self.embedding(inputs), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]

        if self.config.bidirectional:
            # outputs: [max_src_len, batch_size, hidden_size]
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            if self.config.cell == 'gru':
                state = state[:self.config.dec_num_layers]
            else:
                state = (state[0][::2], state[1][::2])

        return outputs, state


class rnn_decoder(nn.Module):

    def __init__(self, config, embedding=None, use_attention=True):
        super(rnn_decoder, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, config.emb_size)
        input_size = 2 * config.emb_size if config.global_emb else config.emb_size 

        if config.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.dec_num_layers, dropout=config.dropout)

        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)

        if not use_attention or config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(config.hidden_size, input_size)
        elif config.attention == 'luong':
            self.attention = models.luong_attention(config.hidden_size, input_size, config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention = models.luong_gate_attention(config.hidden_size, input_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if config.global_emb:
            self.ge_proj1 = nn.Linear(config.emb_size, config.emb_size)
            self.ge_proj2 = nn.Linear(config.emb_size, config.emb_size)
            self.softmax = nn.Softmax(dim=1)

    def forward(self, input, state, output=None, mask=None): 
        embs = self.embedding(input)

        if self.config.global_emb:
            if output is None:
                output = embs.new_zeros(embs.size(0), self.config.tgt_vocab_size)
            probs = self.softmax(output / self.config.tau) 
            emb_avg = torch.matmul(probs, self.embedding.weight)
            H = torch.sigmoid(self.ge_proj1(embs) + self.ge_proj2(emb_avg))
            emb_glb = H * embs + (1 - H) * emb_avg         
            embs = torch.cat((embs, emb_glb), dim=-1)

        output, state = self.rnn(embs, state)
        if self.attention is not None:
            if self.config.attention == 'luong_gate':
                output, attn_weights = self.attention(output)
            else:
                output, attn_weights = self.attention(output, embs)
        else:
            attn_weights = None
        output = self.compute_score(output)

        if self.config.mask and mask:
            mask = torch.stack(mask, dim=1).long()
            output.scatter_(dim=1, index=mask, value=-1e7)

        return output, state, attn_weights

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            lstm = nn.LSTMCell(input_size, hidden_size)
            self.layers.append(lstm)
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1
