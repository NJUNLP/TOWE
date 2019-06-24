

def score_BIO(predicted, golden, ignore_index=-1):
    # B:0, I:1, O:2
    assert len(predicted) == len(golden)
    sum_all = 0
    sum_correct = 0
    golden_01_count = 0
    predict_01_count = 0
    correct_01_count = 0
    # print(predicted)
    # print(golden)
    for i in range(len(golden)):
        length = len(golden[i])
        # print(length)
        # print(predicted[i])
        # print(golden[i])
        golden_01 = 0
        correct_01 = 0
        predict_01 = 0
        predict_items = []
        golden_items = []
        golden_seq = []
        predict_seq = []
        for j in range(length):
            if golden[i][j] == ignore_index:
                break
            if golden[i][j] == 1:
                if len(golden_seq) > 0:  # 00
                    golden_items.append(golden_seq)
                    golden_seq = []
                golden_seq.append(j)
            elif golden[i][j] == 2:
                if len(golden_seq) > 0:
                    golden_seq.append(j)
            elif golden[i][j] == 0:
                if len(golden_seq) > 0:
                    golden_items.append(golden_seq)
                    golden_seq = []
            if predicted[i][j] == 1:
                if len(predict_seq) > 0:  # 00
                    predict_items.append(predict_seq)
                    predict_seq = []
                predict_seq.append(j)
            elif predicted[i][j] == 2:
                if len(predict_seq) > 0:
                    predict_seq.append(j)
            elif predicted[i][j] == 0:
                if len(predict_seq) > 0:
                    predict_items.append(predict_seq)
                    predict_seq = []
        if len(golden_seq) > 0:
            golden_items.append(golden_seq)
        if len(predict_seq) > 0:
            predict_items.append(predict_seq)
        golden_01 = len(golden_items)
        predict_01 = len(predict_items)
        correct_01 = sum([item in golden_items for item in predict_items])
        # print(correct_01)
        # print([item in golden_items for item in predict_items])
        # print(golden_items)
        # print(predict_items)

        golden_01_count += golden_01
        predict_01_count += predict_01
        correct_01_count += correct_01
    precision = correct_01_count/predict_01_count if predict_01_count > 0 else 0
    recall = correct_01_count/golden_01_count if golden_01_count > 0 else 0
    f1 = 2*precision*recall/(precision +recall) if (precision + recall) > 0 else 0
    score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
    return score_dict


def score_aspect(predict_list, true_list):
    correct = 0
    predicted = 0
    relevant = 0

    i = 0
    j = 0
    pairs = []
    while i < len(true_list):
        true_seq = true_list[i]
        predict = predict_list[i]

        for num in range(len(true_seq)):
            if true_seq[num] == 0:
                if num < len(true_seq) - 1:
                    # if true_seq[num + 1] == '0' or true_seq[num + 1] == '1':
                    if true_seq[num + 1] != 1:
                        # if predict[num] == '1':
                        if predict[num] == 0 and predict[num + 1] != 1:
                            # if predict[num] == '1' and predict[num + 1] != '1':
                            correct += 1
                            # predicted += 1
                            relevant += 1
                        else:
                            relevant += 1

                    else:
                        if predict[num] == 0:
                            for j in range(num + 1, len(true_seq)):
                                if true_seq[j] == 1:
                                    if predict[j] == 1 and j < len(predict) - 1:
                                        # if predict[j] == '1' and j < len(predict) - 1:
                                        continue
                                    elif predict[j] == 1 and j == len(predict) - 1:
                                        # elif predict[j] == '1' and j == len(predict) - 1:
                                        correct += 1
                                        relevant += 1

                                    else:
                                        relevant += 1
                                        break

                                else:
                                    if predict[j] != 1:
                                        # if predict[j] != '1':
                                        correct += 1
                                        # predicted += 1
                                        relevant += 1
                                        break


                        else:
                            relevant += 1

                else:
                    if predict[num] == 0:
                        correct += 1
                        # predicted += 1
                        relevant += 1
                    else:
                        relevant += 1

        for num in range(len(predict)):
            if predict[num] == 0:
                predicted += 1

        i += 1

    precision = float(correct) / (predicted + 1e-6)
    recall = float(correct) / (relevant + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1