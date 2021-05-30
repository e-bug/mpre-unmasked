# !/usr/bin/env python

import sys
import argparse
import csv
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
              "cls_prob", "attrs", "classes"]

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 36
MAX_BOXES = 36


def merge_tsvs(fname):
    found_ids = []
    with open(fname) as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
        for item in tqdm(reader):
            img_id = int(item['img_id'])
            found_ids.append(img_id)
    return found_ids

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--split', type=str, default='valid')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()
    args.outfile = "%s_obj%d-%d.tsv" % (args.split, MIN_BOXES, MAX_BOXES)
    
    # Generate TSV files, noramlly do not need to modify
    found_ids = merge_tsvs(args.outfile)
    with open('%s_ids.txt' % args.split, 'w') as f:
        for i in found_ids:
            f.write(str(i) + '\n')

