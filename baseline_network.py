import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_network import TimeEmbedding,TrajEmbedding,Co_Att
from torch_geometric.nn import GCNConv


class FL_FC(nn.Module):
    def __init__(self, embedding_size, date2vec_size, device):
        super(FL_FC, self).__init__()
        self.embedding_size = embedding_size
        self.date2vec_size = date2vec_size
        self.device = device
        dim = date2vec_size
        self.Linear = nn.Linear(dim,dim*2)
        self.Linear2 = nn.Linear(dim*2,dim*2)
        self.layer_norm = nn.LayerNorm(dim*2, eps=1e-6)

    def forward(self, network, traj_seqs, time_seqs):
        batch_size = len(traj_seqs)
        seq_lengths = list(map(len, traj_seqs))
        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)
        s_input = torch.zeros((batch_size, max(seq_lengths), self.embedding_size), dtype=torch.float32)
        t_input = torch.zeros((batch_size, max(seq_lengths), self.date2vec_size), dtype=torch.float32)
        for traj_one in traj_seqs:
            traj_one += [0]*(max(seq_lengths)-len(traj_one))
        for time_one in time_seqs:
            time_one += [[0 for i in range(self.date2vec_size)]] * (max(seq_lengths) - len(time_one))

        traj_seqs = torch.tensor(traj_seqs).to(self.device)
        vec_time_seqs = torch.tensor(time_seqs).to(self.device)

        x = network.x
        for idx, (seq, seqlen) in enumerate(zip(traj_seqs, seq_lengths)):
            s_input[idx, :seqlen] = x.index_select(0, seq[:seqlen])
        for idx, (seq, seqlen) in enumerate(zip(vec_time_seqs, seq_lengths)):
            t_input[idx, :seqlen] = seq[:seqlen]

        seq_lengths = seq_lengths.cpu()
        h=torch.stack([s_input, t_input], 2).to(self.device)
        out_1 = self.Linear(h)
        out_1 = self.Linear2(out_1)
        out = self.layer_norm(out_1)

        s_input = out[:, :, 0, :]
        t_input = out[:, :, 1, :]
        st_input = torch.cat((s_input, t_input), dim=2)
        packed_input = pack_padded_sequence(st_input, seq_lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = pad_packed_sequence(packed_input, batch_first=True)
        outputs = torch.sum(outputs, dim=1)
        return outputs


class FL_RNN(nn.Module):
    def __init__(self, date2vec_size, embedding_size, hidden_size, num_layers, dropout_rate, device):
        super(FL_RNN, self).__init__()
        self.bi_lstm = nn.LSTM(input_size=embedding_size+date2vec_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       batch_first=True,
                       dropout=dropout_rate,
                       bidirectional=True)
        self.device = device
        self.embedding_size = embedding_size
        self.date2vec_size = date2vec_size
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def getMask(self, seq_lengths):
        """
        create mask based on the sentence lengths
        :param seq_lengths: sequence length after `pad_packed_sequence`
        :return: mask (batch_size, max_seq_len)
        """
        max_len = int(seq_lengths.max())

        # (batch_size, max_seq_len)
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0

        return mask

    def forward(self,network, traj_seqs, time_seqs):
        batch_size = len(traj_seqs)
        seq_lengths = list(map(len, traj_seqs))
        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)
        s_input = torch.zeros((batch_size, max(seq_lengths), self.embedding_size), dtype=torch.float32)
        t_input = torch.zeros((batch_size, max(seq_lengths), self.date2vec_size), dtype=torch.float32)
        for traj_one in traj_seqs:
            traj_one += [0]*(max(seq_lengths)-len(traj_one))
        for time_one in time_seqs:
            time_one += [[0 for i in range(self.date2vec_size)]] * (max(seq_lengths) - len(time_one))

        traj_seqs = torch.tensor(traj_seqs).to(self.device)
        vec_time_seqs = torch.tensor(time_seqs).to(self.device)

        x = network.x
        for idx, (seq, seqlen) in enumerate(zip(traj_seqs, seq_lengths)):
            s_input[idx, :seqlen] = x.index_select(0, seq[:seqlen])
        for idx, (seq, seqlen) in enumerate(zip(vec_time_seqs, seq_lengths)):
            t_input[idx, :seqlen] = seq[:seqlen]
        st_input = torch.cat((s_input, t_input), dim=2)
        seq_lengths = seq_lengths.cpu()
        packed_input = pack_padded_sequence(st_input, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bi_lstm(packed_input.to(self.device))  # output, (h, c)
        outputs, seq_lengths = pad_packed_sequence(packed_output, batch_first=True)
        mask = self.getMask(seq_lengths)
        u = torch.tanh(torch.matmul(outputs, self.w_omega))
        # (batch_size, seq_len)
        att = torch.matmul(u, self.u_omega).squeeze()
        att = att.masked_fill(mask == 0, -1e10)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        scored_outputs = outputs * att_score
        outputs = torch.sum(scored_outputs, dim=1)
        return outputs


class FL_GCN(nn.Module):
    def __init__(self, feature_size, date2vec_size, embedding_size, hidden_size,
                                    num_layers, dropout_rate, device):
        super(FL_GCN, self).__init__()
        self.device = device
        self.embedding_S = TrajEmbedding(feature_size, embedding_size, device)
        self.embedding_T = TimeEmbedding(date2vec_size, device)
        self.co_attention = Co_Att(date2vec_size).to(device)
        dim = date2vec_size
        self.Linear = nn.Linear(dim,dim*2)
        self.Linear2 = nn.Linear(dim*2,dim*2)
        self.layer_norm = nn.LayerNorm(dim*2, eps=1e-6)

    def forward(self, network, traj_seqs, time_seqs):
        s_input, seq_lengths = self.embedding_S(network, traj_seqs)
        t_input = self.embedding_T(time_seqs)
        att_s, att_t = self.co_attention(s_input, t_input)

        h=torch.stack([att_s, att_t], 2).to(self.device)
        out_1 = self.Linear(h)
        out_1 = self.Linear2(out_1)
        out = self.layer_norm(out_1)

        s_input = out[:, :, 0, :]
        t_input = out[:, :, 1, :]
        st_input = torch.cat((s_input, t_input), dim=2)
        packed_input = pack_padded_sequence(st_input, seq_lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = pad_packed_sequence(packed_input, batch_first=True)
        outputs = torch.sum(outputs, dim=1)
        return outputs