import codecs
import numpy as np
import pickle
import os
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--embed_path", type=str, default="embedding/glove.840B.300d.txt")
parser.add_argument("--ds", type=str, default="14res")
args = parser.parse_args()
print(args)

glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}


def filter_vocab_embed(ds=None, dim=300):
    # build the vocab for the dataset and filter the embedding table according to the vocabulary.

    f1 = codecs.open(os.path.join('data', ds, "train.tsv"), encoding='utf-8')
    f2 = codecs.open(os.path.join('data', ds, "test.tsv"), encoding='utf-8')
    f_list = [f1, f2]

    all_set = set()
    for f in f_list:
        for line in f:
            line = line.lower().strip('\n').split('\t')[1]
            words = line.split(' ')
            for w in words:
                all_set.add(w)
    word_list = list(all_set)
    word_list.insert(0, '<pad>')
    word_list.append('<unk>')
    vocab_dict = {x: i for i, x in enumerate(word_list)}
    vocab_size = len(word_list)
    print('Vocab size: ', len(word_list))

    embedding_matrix = np.zeros((vocab_size, dim))
    scale = np.sqrt(3.0/dim)
    embedding_matrix[-1] = np.random.uniform(-1*scale, scale, dim)

    hit = 0
    # Filter glove embeddings.
    with codecs.open(args.embed_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f, total=glove_sizes['840B'], desc="Filter glove embeddings"):
            line = line.strip().split(" ")
            word = line[0]
            vector = [float(x) for x in line[1:]]
            if word in vocab_dict:
                word_idx = vocab_dict[word]
                embedding_matrix[word_idx] = np.asarray(vector, dtype='float32')
                hit += 1
    pickle.dump(vocab_dict, open(os.path.join('data', ds, 'vocabulary.pkl'), 'wb'))
    np.save(os.path.join('data', ds, 'embedding_table.npy'), embedding_matrix)
    print("miss:%d" % (vocab_size - hit))


if __name__ == '__main__':
    filter_vocab_embed(args.ds)
