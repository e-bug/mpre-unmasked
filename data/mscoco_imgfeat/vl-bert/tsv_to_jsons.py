import sys
import csv
import json
import os.path
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
              "cls_prob", "attrs", "classes"]


def convert(infiles, outdir):
    for infile in infiles:
        with open(infile, 'r') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for item in tqdm(reader):
                item['img_id'] = int(item['img_id'].split('_')[-1])
                outfile = os.path.join(outdir, str(item['img_id']) + '.json')
                with open(outfile, 'w') as f:
                    json.dump(item, f)

# Train+Val
infiles = ['/data/bugliarello.e/data/mscoco/mscoco_imgfeat/train2014_obj36-36.tsv',
           '/data/bugliarello.e/data/mscoco/mscoco_imgfeat/val2014_obj36-36.tsv']
outdir = '/data/bugliarello.e/data/mscoco/mscoco_imgfeat/vl-bert/trainval_obj36-36'
convert(infiles, outdir)

# Test
infiles = ['/data/bugliarello.e/data/mscoco/mscoco_imgfeat/test2015_obj36-36.tsv']
outdir = '/data/bugliarello.e/data/mscoco/mscoco_imgfeat/vl-bert/test2015_obj36-36'
convert(infiles, outdir)


