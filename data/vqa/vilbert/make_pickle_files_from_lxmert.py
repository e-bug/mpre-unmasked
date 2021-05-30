import _pickle
import json

f = json.load(open('trainval_label2ans.json'))
_pickle.dump(f, open('trainval_label2ans.pkl', 'wb'))

a2l = json.load(open('trainval_ans2label.json'))
_pickle.dump(a2l, open('trainval_ans2label.pkl', 'wb'))

f = json.load(open('train.json'))
fh = []
for e in f:
    fh.append({
        'image_id': int(str(e['question_id'])[:-3]),
        'labels': [a2l[k] for k, v in e['label'].items()],
        'scores': [v for k, v in e['label'].items()],
        'question_id': e['question_id']
    })
_pickle.dump(fh, open('train_target.pkl', 'wb'))

f = json.load(open('minival.json'))
f.extend(json.load(open('nominival.json')))
fh = []
for e in f:
    fh.append({
        'image_id': int(str(e['question_id'])[:-3]),
        'labels': [a2l[k] for k, v in e['label'].items()],
        'scores': [v for k, v in e['label'].items()],
        'question_id': e['question_id']
    })
_pickle.dump(fh, open('val_target.pkl', 'wb'))

