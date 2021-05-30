# coding=utf-8
import sys
import csv
import json
import zlib
import os.path
import pandas as pd

csv.field_size_limit(sys.maxsize)


Split2CapPath = {
    'train': 'Train_GCC-training.tsv',
    'valid': 'Validation_GCC-1.1.0-Validation.tsv'
}
Split2Folder = {
    'train': 'training',
    'valid': 'validation'
}
Split2IdsPath = {
    'train': 'train.clean.ids',
    'valid': 'valid.clean.ids'
}

def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption","url"], usecols=range(0,2))
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df

def _file_name(row):
    return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

def read_captions(corpus_path, split):
    captions = {}
    with open(os.path.join(corpus_path, Split2IdsPath[split])) as f:
        ids = set(l.strip() for l in f.readlines())
    df = open_tsv(os.path.join(corpus_path, Split2CapPath[split]), Split2Folder[split])
    for i, img in enumerate(df.iterrows()):
        caption = img[1]['caption']#.decode("utf8")
        url = img[1]['url']
        im_name = _file_name(img[1])
        image_id = im_name.split('/')[1]
        if image_id in ids:
            captions[image_id] = caption
    return captions

def lxmert_format(captions):
    data = []
    for imgid, caption in captions.items():
        datum = dict()
        datum["img_id"] = imgid
        datum["labelf"] = {}
        datum["sentf"] = {"concap": [caption]}
        data.append(datum)
    return data


if __name__ == '__main__':
    corpus_path = '/data/bugliarello.e/conceptual_captions/'
    for split in ["train", "valid"]:
        captions = read_captions(corpus_path, split)
        with open(os.path.join(corpus_path, "caption_{}.json".format(split)), "w") as f:
            json.dump(captions, f)
        data = lxmert_format(captions)
        with open(os.path.join(corpus_path, "caption_{}.lxmert.json".format(split)), "w") as f:
            json.dump(data, f)

