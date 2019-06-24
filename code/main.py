import codecs
from models import *
from data_helper import load_text_target_label, load_w2v
import os
import pickle
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_bs", type=int, default=8)

parser.add_argument("--EPOCHS", type=int, default=10)
parser.add_argument("--n_hidden", type=int, default=200)
parser.add_argument("--optimizer", type=str,  default="Adam")
parser.add_argument("--model", type=str, default="IOG")
parser.add_argument("--lr", type=float, default=0.2)
parser.add_argument("--freeze", type=bool, default=True)
parser.add_argument("--ds", type=str, default='14res')
parser.add_argument("--l1", type=int, default=1)
parser.add_argument("--l2", type=int, default=1)
parser.add_argument("--pos", type=bool, default=False)
parser.add_argument("--use_char", type=bool, default=False)
parser.add_argument("--use_crf", type=bool, default=False)
parser.add_argument("--projected", type=bool, default=False)
parser.add_argument("--use_elmo", type=bool, default=False)
parser.add_argument("--use_dev", type=bool, default=True)
parser.add_argument("--elmo_mode", type=int, default=6)
parser.add_argument("--elmo_mode2", type=int, default=0)
parser.add_argument("--attn_type", type=int, default=1)
parser.add_argument("--pos_size", type=int, default=30)
parser.add_argument("--char_embed_dim", type=int, default=30)
parser.add_argument("--test", type=int, default=0)
parser.add_argument("--test_model", type=str, default="TCD_LSTM_0.6327.pt")
args = parser.parse_args()

# torch.set_printoptions(profile="full")
print(args)

tag2id = {'B': 1, 'I': 2, 'O': 0}
# seed = 314159
# torch.manual_seed(seed)
# seed = torch.initial_seed()


def main():
    word2index = pickle.load(open(os.path.join('data', args.ds, 'vocabulary.pkl'), "rb"))
    init_embedding = np.load(os.path.join('data', args.ds, 'embedding_table.npy'))
    init_embedding = np.float32(init_embedding)

    # load train data
    print('loading train data...')
    train_text, train_target, train_label = load_text_target_label(os.path.join("data/", args.ds, 'train.tsv'))
    print(train_text[0])
    print(train_target[0])
    print(train_label[0])

    test_text, test_target, test_label = load_text_target_label(os.path.join("data/", args.ds, 'test.tsv'))

    model = NeuralTagger()
    model.train_from_data((train_text, train_target, train_label), (test_text, test_target, test_label), init_embedding, word2index, args)



def test():
    model_name = 'TD_3LSTM_0.7725_0.8020_14res2.pt'
    # f_test = "data/%s/test_all.txt" % ds
    f_test = "polarity_level/data_aspect/{0}/test/term_all.txt".format('res')
    test_text, test_t, test_ow = load_data(filename=f_test)
    f_w2v = "data/%s/embedding_all_glove300.txt" % ds
    W, word2index = load_w2v(f_w2v)
    model = NeuralTagger_elmo()
    rnn = torch.load("backup3/%s" % model_name)
    result = model.predict(rnn, (test_text, test_t, test_ow), word2index, args)
    test_file = "case_study/" + model_name[0:-3] + "_test.txt"
    fw = codecs.open(test_file, 'w', encoding='utf-8')
    fw2 = codecs.open("data_aspect/{0}/test/ow.txt".format('res'), 'w', encoding='utf-8')
    # print(result)
    assert len(result) == len(test_text)
    for s, t, p, g in zip(test_text, test_t, result, test_ow):
        t = ' '.join([str(i) for i in t])
        p = p.tolist()
        p = ' '.join([str(i) for i in p])
        # print(p)
        # print(g)
        g = ' '.join([str(i) for i in g])
        fw.write(' '.join(s) + '\t' + t + '\t' + p + '\t' + g + '\t' + str(p==g) + '\n')
        fw2.write(p+'\n')


if __name__ == '__main__':
    if args.test == 0:
        main()
    else:
        test()
