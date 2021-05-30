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
                outfile = os.path.join(outdir, str(item['img_id']) + '.json')
                with open(outfile, 'w') as f:
                    json.dump(item, f)

infiles = ['/data/bugliarello.e/data/vcr/vcr_gt_imgfeat/vcr_gt_obj36-36.tsv']
outdir = '/data/bugliarello.e/data/vcr/vcr_gt_imgfeat/vl-bert/vcr_gt_obj36-36'
convert(infiles, outdir)

