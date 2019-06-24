import torch
from torch import nn
import time
import numpy as np
import torch.nn.functional as F
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import score_BIO

cuda_flag = True and torch.cuda.is_available()


def train(model, train_iter, dev_iter, test_iter, args):
    # Train:
    # HyperParameter
    n_epochs = args.EPOCHS
    learning_rate = args.lr

    n_gpu = torch.cuda.device_count()
    print("CUDA: " +str(cuda_flag))
    if cuda_flag:
        model = model.cuda()
    # rnn = TextCNN(300, output_size, max_length)
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters())
    if args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters())
    if args.freeze:
        for param in model.word_rep.word_embed.parameters():
            param.requires_grad = False
    print_every = 100
    plot_every = 30
    # Keep track of losses for plotting
    current_loss = []
    all_losses = []
    test_acc = []
    acc_index = 0
    es_len = 0

    start = time.time()
    best_score = 0

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # np.random.seed([3, 1415])

    for epoch in range(int(n_epochs)):
        loss_sum = 0
        tr_loss = 0
        tr_loss_avi = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_iter):

            model.train()
            output_p = model(batch)
            loss = 0
            if model.use_crf:
                loss = model.crf._neg_log_likelihood(output_p, batch.label)
            else:
                # for output, label in zip(output_p, batch.label):
                #     loss += F.nll_loss(output, label, ignore_index=-1)
                loss = F.nll_loss(output_p.view(-1, 3), batch.label.view(-1), ignore_index=-1)
            if n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            model.zero_grad()

            current_loss.append(loss.item())
            loss_sum += loss.item()

            tr_loss += loss.item()
            tr_loss_avi += loss.item() * batch.batch_size
            nb_tr_examples += batch.batch_size
            nb_tr_steps += 1

            if (step+1) % plot_every == 0:
                # print(batch_epoch)
                all_losses.append(sum(current_loss) / len(current_loss))
                current_loss = []

        with torch.no_grad():
            eval_dict = eval(model, dev_iter, args)
        print("Epoch:%d" % epoch)
        print("DEV: p:%.4f, r:%.4f, f:%.4f" % (eval_dict['precision'], eval_dict['recall'], eval_dict['f1']))
        with torch.no_grad():
            test_dict = eval(model, test_iter, args)
        print("TEST: p:%.4f, r:%.4f, f:%.4f" % (test_dict['precision'], test_dict['recall'], test_dict['f1']))
        # print("OPINION: p:%.4f, r:%.4f, f:%.4f" % (pred_acc_t2[0], pred_acc_t2[1], pred_acc_t2[2]))

        if eval_dict['main_metric'] > best_score:
            best_score = eval_dict['main_metric']
            max_print = ("Epoch%d\n" % epoch
                         + "DEV: p:%.4f, r:%.4f, f:%.4f\n" % (eval_dict['precision'], eval_dict['recall'], eval_dict['f1'])
                         + "TEST: p:%.4f, r:%.4f, f:%.4f\n" % (test_dict['precision'], test_dict['recall'], test_dict['f1']))
            best_model = "backup/%s_%.4f_%.4f.pt" % (args.model, best_score, test_dict['main_metric'])
            best_dict = copy.deepcopy(model)

        print("Epoch: %d, loss: %.4f" % (epoch, loss_sum))

    torch.save(best_dict, best_model)
    print("Best Result:")
    print(max_print)
    # plt.figure()
    # plt.plot(all_losses)
    time_stamp = time.asctime().replace(':', '_').split()
    print(time_stamp)
    # plt.savefig("fig/foor_%s.png" % '_'.join(time_stamp))
    # plt.show()
    return best_dict


def category_from_output(output):

    top_n, top_i = output.topk(1) # Tensor out of Variable with .data
    # print(top_i)
    category_i = top_i.view(output.size()[0], -1).detach().cpu().numpy().tolist()
    return category_i

def eval(model, dev_iter, args):
    model.eval()

    logit_list = []
    labels_eval_list = []
    predicted_result = []
    for eval_batch in dev_iter:
        output = model(eval_batch)
        if args.use_crf:
            _, category_i_p = model.crf._viterbi_decode(output)
        else:
            category_i_p = category_from_output(output)
        # print(eval_batch.target)
        # print(output)
        # print(category_i_p)
        predicted_result.extend(category_i_p)
        label_ids = eval_batch.label.to('cpu').numpy()
        labels_eval_list.extend(label_ids.tolist())
        logits = output.detach().cpu().numpy()
        logit_list.extend(logits.tolist())

    eval_dict = {}

    score_dict = score_BIO(predicted_result, labels_eval_list, ignore_index=-1)

    eval_dict.update(score_dict)
    eval_dict['main_metric'] = score_dict['f1']
    return eval_dict
