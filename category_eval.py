import json
import argparse


def main(pred_file):
    with open('CoQA/dev.json', 'r') as f:
        data = json.load(f)['data']
    gold_answes = []
    for article in data:
        questions = article['questions']
        answers = article['answers']
        for i, (question, answer) in enumerate(zip(questions, answers)):
            id = article['id']
            turn_id = question['turn_id']
            gold_answes.append({"id": id, "turn_id": turn_id, "answer": answer['input_text']})

    with open(pred_file, 'r') as f:
        preds = json.load(f)

    eva = []
    precision = []
    recall = []
    for i in range(4):
        a = []
        for j in range(4):
            a.append(0)
        eva.append(a)
    # 0:unknown 1:yes 2:no 3:x
    for i, (pred, gold) in enumerate(zip(preds, gold_answes)):
        pred = pred['answer'].strip().lower()
        gold = gold['answer'].strip().lower()
        a = judge(gold)
        b = judge(pred)
        eva[a][b] += 1
    for i in range(4):
        tp = float(eva[i][i])
        fp = 0.0
        for j in range(4):
            if j != i:
                fp += eva[j][i]
        fn = 0.0
        for j in range(4):
            if j != i:
                fn += eva[i][j]
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))
    yes = {'precision': precision[1], 'recall': recall[1]}
    no = {'precision': precision[2], 'recall': recall[2]}
    x = {'precision': precision[3], 'recall': recall[3]}
    unknown = {'precision': precision[0], 'recall': recall[0]}
    output = {'yes': yes, 'no': no, 'x': x, 'unknown': unknown}
    print(json.dumps(output, indent=4))


def judge(answer):
    if answer == 'yes':
        return 1
    if answer == 'no':
        return 2
    if answer == 'unknown':
        return 0
    return 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser('description: experiments on datasets')
    parser.add_argument('pred_file')
    args = parser.parse_args()
    main(args.pred_file)
