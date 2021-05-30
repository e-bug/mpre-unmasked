# !/usr/bin/env python

import os.path
import sys
import argparse
import csv
import json
import numpy as np
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
              "cls_prob", "attrs", "classes"]

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 36
MAX_BOXES = 36


def merge_tsvs(infile, outfolder, min_ix, max_ix):

    # Init with test files
    found_ids = {2963869831, 3866049978, 3769223247, 2661358740, 2166031423, 4178943787, 2672943342, 2618562984,
                 1426359054, 1612953290, 2855930558, 3596596878, 4041895454, 1520047054, 3133675427, 2525648016}

    with open(infile) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for ix, item in enumerate(tqdm(reader)):
            if ix < min_ix:
               continue
            if ix > max_ix:
               break
            img_id = int(item['img_id'])
            if img_id not in found_ids:
                try:
                    json_file = "{}.json".format(img_id)
                    with open(os.path.join(outfolder, json_file), 'w') as f:
                        json.dump(item, f)
                    found_ids.add(img_id)
                except Exception as e:
                    print e


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--total_group', type=int, default=1,
                        help="the number of group for exracting")
    parser.add_argument('--group_id', type=int, default=0,
                        help=" group id for current analysis, used to shard")
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()
    args.infile = "%s_obj%d-%d.tsv" % (args.split, MIN_BOXES, MAX_BOXES)
    args.outfolder = "/data/bugliarello.e/conceptual_captions/%s_obj%d-%d.clean" % (args.split, MIN_BOXES, MAX_BOXES)

    if args.split == 'valid':
        upper_range = np.arange(15000)
    else:
        upper_range = np.arange(2850000)
    
    ixs = np.array_split(upper_range, args.total_group)[args.group_id]
    min_ix, max_ix = ixs[0], ixs[-1]

    # Generate TSV files, noramlly do not need to modify
    merge_tsvs(args.infile, args.outfolder, min_ix, max_ix)

