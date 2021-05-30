import zlib

def _file_name(url):
    return "%s" % (zlib.crc32(url.encode('utf-8')) & 0xffffffff)

BASEDIR = "/data/bugliarello.e/conceptual_captions/"
captions = []
urls = []

with open(BASEDIR + 'Train_GCC-training.tsv') as fp:
    for cnt, line in enumerate(fp):
        s = line.split('\t')
        captions.append(s[0].split(' '))
        urls.append(s[1][:-1])

valids = set([])
with open(BASEDIR + 'train.clean.ids') as fp:
    for cnt, line in enumerate(fp):
        valids.add(line[:-1])

import json
with open(BASEDIR + 'vl-bert/train.json', 'w') as outfile:
    for (cap, url) in zip(captions, urls):
        im = _file_name(url)
        if (im in valids):
            d = {'image':"train_image.zip@/{}".format(im), 'caption':cap}
            json.dump(d, outfile)
            outfile.write('\n')

import json
with open(BASEDIR + 'vl-bert/train_frcnn.json', 'w') as outfile:
    for (cap, url) in zip(captions, urls):
        im = _file_name(url)
        if (im in valids):
            d = {'image':"train_image.zip@/{}".format(im), 'caption':cap, 'frcnn':"train_frcnn.zip@/{}.json".format(im)}
            json.dump(d, outfile)
            outfile.write('\n')

