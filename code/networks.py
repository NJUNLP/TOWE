import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class WordRep(nn.Module):
    def __init__(self, vocab_size, word_embed_dim, char_size, args):
        super(WordRep, self).__init__()
        self.use_char = args.use_char
        self.use_elmo = args.use_elmo
        self.elmo_mode = args.elmo_mode
        self.elmo_mode2 = args.elmo_mode2
        self.projected = args.projected
        self.char_embed_dim = args.char_embed_dim
        self.word_embed = nn.Embedding(vocab_size, word_embed_dim)
        if self.use_elmo:
            self.elmo_weights = nn.Linear(3, 1)
            self.elmo_proj = nn.Linear(1024, word_embed_dim)
        if self.use_char:
            self.char_embed = nn.Embedding(char_size, self.char_embed_dim)
            self.char_lstm = nn.LSTM(self.char_embed_dim, self.char_embed_dim//2, num_layers=1, bidirectional=True)

    def forward(self, batch):
        sentence, _ = batch.text
        # sentence = torch.unsqueeze(sentence, -1)
        # print("sentence in wordrep")
        # print(sentence)

        # sentence = sentence.view(sentence.size()[1], -1)
        if self.use_elmo:
            elmo_tensor = batch.elmo
        else:
            elmo_tensor = None
        char_seq = None
        char_seq_len = None
        char_seq_recover = None
        words_embeds = self.word_embed(sentence)
        if self.use_elmo:
            if self.elmo_mode == 2:
                elmo_tensor = elmo_tensor[-1]
            elif self.elmo_mode == 3:
                elmo_tensor = elmo_tensor[1]
            elif self.elmo_mode == 4:
                elmo_tensor = elmo_tensor[0]
            elif self.elmo_mode == 6:
                attn_weights = F.softmax(self.elmo_weights.weight, dim=-1)
                elmo_tensor = torch.matmul(attn_weights, elmo_tensor.t())
            else:
                elmo_tensor = elmo_tensor.mean(dim=0)
            if not self.projected:
                projected = elmo_tensor
            else:
                projected = self.elmo_proj(elmo_tensor)
            # print(words_embeds.size())
            # exit(-1)
            projected = projected.view(projected.size()[0], 1, -1)
            if self.elmo_mode2 == 1:
                words_embeds = words_embeds + projected
            elif self.elmo_mode2 == 2:
                words_embeds = words_embeds
            elif self.elmo_mode2 == 3:
                words_embeds = torch.cat((words_embeds, projected), dim=-1)
            else:
                words_embeds = projected
        if self.use_char:
            char_embeds = self.char_embed(char_seq)
            pack_seq = pack_padded_sequence(char_embeds, char_seq_len, True)
            char_rnn_out, char_hidden = self.char_lstm(pack_seq)
            last_hidden = char_hidden[0].view(sentence.size()[0], 1, -1)
            # print(words_embeds)
            # print(last_hidden)
            words_embeds = torch.cat((words_embeds, last_hidden), -1)
        return words_embeds


class IOG(nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, args=None):
        super(IOG, self).__init__()

        l1 = 1 if args is None else args.l1
        l2 = 1 if args is None else args.l2
        self.use_crf = args.use_crf if args is not None else False
        self.input_size = word_embed_dim + args.char_embed_dim if args.use_char else word_embed_dim
        if args.elmo_mode2 == 3 and args.projected and args.use_elmo:
            self.input_size += word_embed_dim
        if args.elmo_mode2 == 0 and not args.projected and args.use_elmo:
            self.input_size = 1024
        if args.elmo_mode2 == 3 and not args.projected and args.use_elmo:
            self.input_size += 1024
        self.hidden_size = args.n_hidden
        self.output_size = output_size+2 if self.use_crf else output_size
        self.max_length = 1

        self.word_rep = WordRep(vocab_size, word_embed_dim, None, args)

        self.rnn_L = nn.LSTM(self.input_size, self.hidden_size, num_layers=l1, bidirectional=True, batch_first=True)
        self.rnn_R = nn.LSTM(self.input_size, self.hidden_size, num_layers=l1, bidirectional=True, batch_first=True)
        self.rnn_global = nn.LSTM(self.input_size, self.hidden_size, num_layers=l1, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(self.hidden_size*4, self.output_size)
        self.dropout = nn.Dropout(0.0)
        if self.use_crf:
            self.crf = CRF(self.output_size)

    def forward(self, batch):
        target = batch.target
        sentence = self.word_rep(batch)

        left_mask = batch.left_mask
        right_mask = batch.right_mask
        target_mask = target != 0

        left_context = sentence * left_mask.unsqueeze(-1).float().expand_as(sentence)
        right_context = sentence * right_mask.unsqueeze(-1).float().expand_as(sentence)

        left_encoded, _ = self.rnn_L(left_context)
        right_encoded, _ = self.rnn_R(right_context)
        global_encoded, _ = self.rnn_global(sentence)

        left_encoded = left_encoded * left_mask.unsqueeze(-1).float().expand_as(left_encoded)
        right_encoded = right_encoded * right_mask.unsqueeze(-1).float().expand_as(right_encoded)

        encoded = left_encoded + right_encoded
        target_average_mask = 1 - 1/2*target_mask.unsqueeze(-1).float().expand_as(encoded)
        encoded = encoded * target_average_mask

        encoded = torch.cat((encoded, global_encoded), dim=-1)
        # print(encoded)
        decodedP = self.fc(encoded)

        outputP = F.log_softmax(decodedP, dim=-1)
        return outputP


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--EPOCHS", type=int, default=20)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--model", type=str, default="TCB_LSTM")
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--ds", type=str, default='14res')
    parser.add_argument("--l1", type=int, default=2)
    parser.add_argument("--l2", type=int, default=2)
    parser.add_argument("--pos", type=bool, default=False)
    parser.add_argument("--use_char", type=bool, default=False)
    parser.add_argument("--use_crf", type=bool, default=False)
    parser.add_argument("--projected", type=bool, default=False)
    parser.add_argument("--use_elmo", type=bool, default=False)
    parser.add_argument("--elmo_mode", type=int, default=6)
    parser.add_argument("--elmo_mode2", type=int, default=2)
    parser.add_argument("--pos_size", type=int, default=30)
    parser.add_argument("--char_embed_dim", type=int, default=30)
    parser.add_argument("--attn_type", type=int, default=1)
    args = parser.parse_args()

    inputs = torch.LongTensor([1, 2, 3, 10]).view(4, -1)
    target = torch.LongTensor([0, 1, 2, 0])
    char_seq_tensor = torch.LongTensor([[3, 4, 5, 6, 7], [4, 5, 6, 7, 0], [1, 2, 3, 0, 0], [3, 5, 0, 0, 0]])
    char_len_tensor = torch.LongTensor([5, 4, 3, 2])
    char_recover = torch.LongTensor([3, 2, 1, 0])
    elmo_tensor = torch.randn([2,4,5])
    print(elmo_tensor)
    model = GCAE_LSTM(6, 3, 100, args)
    print(model((inputs, elmo_tensor), target))
