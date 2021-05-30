import _pickle
import json

INDIR = '/data/bugliarello.e/data/gqa/lxmert'
OUTDIR = '/data/bugliarello.e/data/gqa/cache'

f = json.load(open(INDIR + '/trainval_label2ans.json'))
_pickle.dump(f, open(OUTDIR + '/trainval_label2ans.pkl', 'wb'))

a2l = json.load(open(INDIR + '/trainval_ans2label.json'))
_pickle.dump(a2l, open(OUTDIR + '/trainval_ans2label.pkl', 'wb'))

f = json.load(open(INDIR + '/train.json'))
fht = []
for e in f:
    fht.append({'image_id': int(e['img_id']), 'labels': [a2l[k] for k,v in e['label'].items()], 'scores': [v for k,v in e['label'].items()], 'question_id': e['question_id'], 'question': e['sent']})
_pickle.dump(fht, open(OUTDIR + '/train_target.pkl', 'wb'))

f = json.load(open(INDIR + '/valid.json'))
fhv = []
for e in f:
    fhv.append({'image_id': int(e['img_id']), 'labels': [a2l[k] for k,v in e['label'].items()], 'scores': [v for k,v in e['label'].items()], 'question_id': e['question_id'], 'question': e['sent']})
_pickle.dump(fhv, open(OUTDIR + '/val_target.pkl', 'wb'))

fh = fht + fhv
_pickle.dump(fh, open(OUTDIR + '/trainval_target.pkl', 'wb'))

