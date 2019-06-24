import codecs
import numpy as np

def process_files():
    from gensim.models.keyedvectors import KeyedVectors

    my_model = KeyedVectors.load_word2vec_format('embedding_all_GN300.txt')
    f_train1 = codecs.open("dataset/train_docs.txt", encoding='utf-8')
    f_train2 = open("dataset/train_labels_a.txt")
    f_train3 = open("dataset/train_labels_p.txt")
    f_test1 = codecs.open("dataset/test_docs.txt", encoding='utf-8')
    f_test2 = open("dataset/test_labels_a.txt")
    f_test3 = open("dataset/test_labels_p.txt")

    train_x_text = [line.strip().lower() for line in f_train1]
    print('calm')
    train_y_t = [[int(i) for i in line.strip('\r\n').split()] for line in f_train2]
    train_y_p = [[int(i) for i in line.strip('\r\n').split()] for line in f_train3]
    dev_x_text = [line.strip().lower() for line in f_test1]
    dev_y_t = [[int(i) for i in line.strip('\r\n').split()] for line in f_test2]
    dev_y_p = [[int(i) for i in line.strip('\r\n').split()] for line in f_test3]


    train_feature_x = [line_to_tensor(s.split(' '), my_model) for s in train_x_text]  # [L*1*dim]
    dev_feature_x = [line_to_tensor(s.split(' '), my_model) for s in dev_x_text]

    for i in range(len(dev_y_t)):
        if not len(dev_y_t[i]) == len(dev_feature_x[i]):
            print(dev_y_t[i])
            print(i)
            print(dev_x_text[i])

    pickle.dump(train_feature_x, open('data/train_x_f.pkl', 'wb'))
    pickle.dump(dev_feature_x, open('data/dev_x_f.pkl', 'wb'))
    pickle.dump(train_y_t, open('data/train_y_t.pkl', 'wb'))
    pickle.dump(train_y_p, open('data/train_y_p.pkl', 'wb'))
    pickle.dump(dev_y_t, open('data/dev_y_t.pkl', 'wb'))
    pickle.dump(dev_y_p, open('data/dev_y_p.pkl', 'wb'))


def line_to_index(words, word2index):
    # size = model.vector_size
    # tensor = torch.zeros(1, 1, max_length, size)
    index_list = []
    # print(words)
    # print(len(words))
    for li, w in enumerate(words):
        # if re.findall(r"[A-Za-z0-9]", w):
        #     w = w.translate(str.maketrans('', '', string.punctuation))
        # if '-' in w:
        #     ws = w.split('-')
        #     for i in range(len(ws)):
        #         words.extend(ws)
        if w in word2index:
            vec = word2index[w]
            # vec_tensor = torch.from_numpy(vec)
        else:
            print(w)
            print('error')
            exit(-1)
        index_list.append(vec)

    if len(index_list) == 0:
        # print(words)
        return torch.zeros(1, 1)
    length = len(index_list)
    feature_array = np.asarray(index_list)
    # print(feature_array)
    # if padding:
    #     feature_array = np.lib.pad(feature_array, ((0, max_length - len(feature_array)), (0, 0)), 'constant')
    # print(feature_array)
    tensor = torch.from_numpy(feature_array)  # N*d
    # print(tensor)
    tensor = tensor.view(tensor.size()[0], -1).long()  # 1*1*max_length*dim
    # print(tensor.shape)
    # exit(0)
    return tensor


def load_w2v(filename):
    f_w2v = codecs.open(filename, encoding="utf-8").readlines()
    vocab_size = int(f_w2v[0].split()[0])
    embed_size = int(f_w2v[0].split()[1])
    print("Vocab size: %d" % vocab_size)
    print("Embed size: %d" % embed_size)
    word2index = dict()
    W = np.zeros(shape=(vocab_size, embed_size), dtype='float32')
    for i in range(1, vocab_size+1):
        line = f_w2v[i].strip('\r\n').split()
        w = line[0]
        vec = line[1:]
        vec = [float(v) for v in vec]
        assert len(vec) == embed_size
        word2index[w] = i - 1
        W[i-1] = vec
    # print(len(W))
    # print(W[0])
    return W, word2index


def load_data(filename):
    f_test = codecs.open(filename, encoding='utf-8')
    test_text = []
    test_t = []
    test_ow = []
    for line in f_test:
        text, target_labels, ow_labels = line.strip().split('\t')
        # text = text.lower()
        text = text.split(' ')
        target_labels = [int(i) for i in target_labels.split()]
        ow_labels = [int(i) for i in ow_labels.split()]
        assert len(target_labels) == len(text)
        test_text.append(text)
        test_t.append(target_labels)
        test_ow.append(ow_labels)
    return test_text, test_t, test_ow


def load_text_target_label(path):
    text_list = []
    target_list = []
    label_list = []
    with codecs.open(path, encoding='utf-8') as fo:
        for i, line in enumerate(fo):
            if i == 0:
                continue
            s_id, sentence, target_tags, opinion_words_tags = line.split('\t')
            text_list.append(sentence.strip())
            w_t = target_tags.strip().split(' ')
            target = [t.split('\\')[-1] for t in w_t]
            target_list.append(target)
            w_l = opinion_words_tags.strip().split(' ')
            label = [l.split('\\')[-1] for l in w_l]
            label_list.append(label)
    return text_list, target_list, label_list



def generate_sentence_label(train_texts, train_ow):  # combine all ow labels for one sentence
    train_s_texts = []
    train_s_ow = []
    prev_text = ''
    train_s_t = []
    for i in range(len(train_texts)):
        if train_texts[i] != prev_text:
            prev_text = train_texts[i]
            train_s_texts.append(train_texts[i])
            train_s_ow.append([train_ow[i]])
        else:
            train_s_ow[-1].append(train_ow[i])
    print(len(train_s_texts))
    new_s_ow = []
    for t, o in zip(train_s_texts, train_s_ow):
        train_s_t.append([0 for i in range(len(t))])
        oarray = np.asarray(o)
        new_ow = oarray.max(axis=0).tolist()
        new_s_ow.append(new_ow)
        # print(str(t)+'\t'+str(o) + '\t' + str(new_ow))
    return train_s_texts, new_s_ow, new_s_ow


if __name__ == '__main__':
    load_pos('14res')
