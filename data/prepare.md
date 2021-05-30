# Data prep
DATADIR=/home/pmh864/data

## Img feats
1. sudo docker pull airsplay/bottom-up-attention
2. wget 'https://www.dropbox.com/s/nu6jwhc88ujbw1v/resnet101_faster_rcnn_final_iter_320000.caffemodel?dl=1' -O snap/pretrained/resnet101_faster_rcnn_final_iter_320000.caffemodel
3. wget 'https://www.dropbox.com/s/wqada4qiv1dz9dk/resnet101_faster_rcnn_final.caffemodel?dl=1' -O snap/pretrained/resnet101_faster_rcnn_final.caffemodel

### MS COCO
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip

cp $(pwd)/data/mscoco_imgfeat/extract_coco_image.py /data/bugliarello.e/data/mscoco/mscoco_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/mscoco/images:/workspace/images:ro -v /data/bugliarello.e/data/mscoco/mscoco_imgfeat:/workspace/features -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=6 python extract_coco_image.py --split train --group_id 0 --total_group 4
CUDA_VISIBLE_DEVICES=7 python extract_coco_image.py --split train --group_id 1 --total_group 4
CUDA_VISIBLE_DEVICES=8 python extract_coco_image.py --split train --group_id 2 --total_group 4
CUDA_VISIBLE_DEVICES=9 python extract_coco_image.py --split train --group_id 3 --total_group 4


### Flickr30k
wget --header="Host: uc43ec8673cf2ea8eaee7f788907.dl.dropboxusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,it;q=0.8,ja;q=0.7" --header="Referer: https://www.dropbox.com/" "https://uc43ec8673cf2ea8eaee7f788907.dl.dropboxusercontent.com/cd/0/get/A4NnaJaicC7_18R6YOLXWlLUGUHftMPwhNEpvQsCFSN3_cmbsLGpCFeoM2DIjJYGeXQtNN5knM5RVexHZJ8ag3gSCrKbihxAtuT-k0eol9nSRB5GtM-cbV9nc2g8WLAfEQo/file?_download_id=064031497742737241522600926201303285326862305832269827186878902414&_notify_domain=www.dropbox.com&dl=1" -c -O 'hard_negative.pkl'

cp $(pwd)/data/flickr30k_imgfeat/extract_flickr30k_image.py /data/bugliarello.e/data/flickr30k/flickr30k_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/flickr30k:/workspace/images:ro -v /data/bugliarello.e/data/flickr30k/flickr30k_imgfeat:/workspace/features -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=3 python extract_flickr30k_image.py --split test --group_id 0 --total_group 1
CUDA_VISIBLE_DEVICES=4 python extract_flickr30k_image.py --split valid --group_id 0 --total_group 1
CUDA_VISIBLE_DEVICES=5 python extract_flickr30k_image.py --split train --group_id 0 --total_group 5
CUDA_VISIBLE_DEVICES=6 python extract_flickr30k_image.py --split train --group_id 1 --total_group 5
CUDA_VISIBLE_DEVICES=7 python extract_flickr30k_image.py --split train --group_id 2 --total_group 5
CUDA_VISIBLE_DEVICES=8 python extract_flickr30k_image.py --split train --group_id 3 --total_group 5
CUDA_VISIBLE_DEVICES=9 python extract_flickr30k_image.py --split train --group_id 4 --total_group 5

CUDA_VISIBLE_DEVICES=5 python extract_flickr30k_image.py --split flickr30k --group_id 0 --total_group 5


### VQA
VL-BERT:
wget --header="Host: doc-0c-a4-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,it;q=0.8,ja;q=0.7" --header="Referer: https://drive.google.com/" --header="Cookie: AUTH_f4k8g0tslhvcm6fmv2c5eu98j4d8ropj_nonce=304rc382d91o2" --header="Connection: keep-alive" "https://doc-0c-a4-docs.googleusercontent.com/docs/securesc/cfnd6k8evm1fc2hgma7tsh3it4pm1ksf/jv6loairmtuddlqkq60olqkp5unhg1eg/1592067000000/15420328229527158211/07893326812593803856/1CPnYcOgIOP5CZkp_KChuCg54_Ljr6-fp?e=download&authuser=0&nonce=304rc382d91o2&user=07893326812593803856&hash=n960rblra74dnbpvcnced7nsvs0jaosp" -c -O 'answers_vqa.txt'

### GQA (incl. VG)
cp $(pwd)/data/vg_gqa_imgfeat/extract_gqa_image.py /data/bugliarello.e/data/gqa/vg_gqa_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/gqa/images:/workspace/images:ro -v /data/bugliarello.e/data/gqa/vg_gqa_imgfeat:/workspace/features -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=5 python extract_gqa_image.py --group_id 0 --total_group 5


### VCR
wget --header="Host: uc130210cb613274b30552126138.dl.dropboxusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,it;q=0.8,ja;q=0.7" --header="Referer: https://www.dropbox.com/" "https://uc130210cb613274b30552126138.dl.dropboxusercontent.com/cd/0/get/A4glmz-ASU-btrQzFmsJ9_58Wq7ArUrLGup6iiHaUXjnJyZxAopSff0tPCkPvFM_zTfNsTTWus9a-FCIdv1DPnBna4lB_kJCIs2YA5x3eeQfFJdMwOC1gKvKy2Fvfjs8Tuw/file?_download_id=07628472918562257444061356783152624733726562439249235261243655781&_notify_domain=www.dropbox.com&dl=1" -c -O 'unisex_names_table.csv' -O /data/bugliarello.e/data/vcr/unisex_names_table.csv

mkdir -p /data/bugliarello.e/data/vcr/vcr_imgfeat
cp $(pwd)/data/vcr_imgfeat/extract_vcr_image.py /data/bugliarello.e/data/vcr/vcr_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/vcr/vcr1images:/workspace/images:ro -v /data/bugliarello.e/data/vcr/vcr_imgfeat:/workspace/features -v /data/bugliarello.e/data/vcr/annotations:/workspace/annotations:ro -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
pip install json-lines
cd /workspace/features
CUDA_VISIBLE_DEVICES=5 python extract_vcr_image.py --group_id 0 --total_group 5

mkdir -p /data/bugliarello.e/data/vcr/vcr_gt_imgfeat
cp $(pwd)/data/vcr_imgfeat/extract_vcr_gt_image.py /data/bugliarello.e/data/vcr/vcr_gt_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/vcr/vcr1images:/workspace/images:ro -v /data/bugliarello.e/data/vcr/vcr_gt_imgfeat:/workspace/features -v /data/bugliarello.e/data/vcr/annotations:/workspace/annotations:ro -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
pip install json-lines
cd /workspace/features
CUDA_VISIBLE_DEVICES=5 python extract_vcr_gt_image.py --group_id 0 --total_group 5


### Refcoco_unc (all images from mscoco train2014)
p = pickle.load(open('../data/refcoco/refs(unc).p', 'rb'))
with open('/data/bugliarello.e/data/refer/refcoco_unc.json', 'w') as f:
     json.dump(p, f)

wget http://bvision.cs.unc.edu/licheng/MattNet/detections.zip -P /data/bugliarello.e/data/refer
mkdir -p /data/bugliarello.e/data/refer/refcoco_unc_imgfeat
cp $(pwd)/data/refcoco_unc_imgfeat/extract_refcoco_unc_image.py /data/bugliarello.e/data/refer/refcoco_unc_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/mscoco/images/train2014:/workspace/images:ro -v /data/bugliarello.e/data/refer/refcoco_unc_imgfeat:/workspace/features -v /data/bugliarello.e/data/refer:/workspace/annotations:ro -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=5 python extract_refcoco_unc_image.py --group_id 0 --total_group 1

mkdir -p /data/bugliarello.e/data/refer/refcoco_unc_gt_imgfeat
cp $(pwd)/data/refcoco_unc_imgfeat/extract_refcoco_unc_gt_image.py /data/bugliarello.e/data/refer/refcoco_unc_gt_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/mscoco/images/train2014:/workspace/images:ro -v /data/bugliarello.e/data/refer/refcoco_unc_gt_imgfeat:/workspace/features -v /data/bugliarello.e/data/refer:/workspace/annotations:ro -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=5 python extract_refcoco_unc_gt_image.py --group_id 0 --total_group 1

cd vilbert/tools/refer; make
mkdir -p /gs/hs0/tgb-deepmt/bugliarello.e/data/refer/data/cache

### Refcoco+_unc (all images from mscoco train2014)
mkdir -p /data/bugliarello.e/data/refer/refcoco+_unc_imgfeat
cp $(pwd)/data/refcoco+_unc_imgfeat/extract_refcoco+_unc_image.py /data/bugliarello.e/data/refer/refcoco+_unc_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/mscoco/images/train2014:/workspace/images:ro -v /data/bugliarello.e/data/refer/refcoco+_unc_imgfeat:/workspace/features -v /data/bugliarello.e/data/refer:/workspace/annotations:ro -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=5 python extract_refcoco+_unc_gt_image.py --group_id 0 --total_group 1

mkdir -p /data/bugliarello.e/data/refer/refcoco+_unc_gt_imgfeat
cp $(pwd)/data/refcoco+_unc_imgfeat/extract_refcoco+_unc_gt_image.py /data/bugliarello.e/data/refer/refcoco+_unc_gt_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/mscoco/images/train2014:/workspace/images:ro -v /data/bugliarello.e/data/refer/refcoco+_unc_gt_imgfeat:/workspace/features -v /data/bugliarello.e/data/refer:/workspace/annotations:ro -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=5 python extract_refcoco_unc+_gt_image.py --group_id 0 --total_group 1


### Refcocog_umd (all images from mscoco train2014)
mkdir -p /data/bugliarello.e/data/refer/refcocog_umd_imgfeat
cp $(pwd)/data/refcocog_umd_imgfeat/extract_refcocog_umd_image.py /data/bugliarello.e/data/refer/refcocog_umd_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/mscoco/images/train2014:/workspace/images:ro -v /data/bugliarello.e/data/refer/refcocog_umd_imgfeat:/workspace/features -v /data/bugliarello.e/data/refer:/workspace/annotations:ro -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=5 python extract_refcocog_umd_image.py --group_id 0 --total_group 1

mkdir -p /data/bugliarello.e/data/refer/refcocog_umd_gt_imgfeat
cp $(pwd)/data/refcocog_umd_imgfeat/extract_refcocog_umd_gt_image.py /data/bugliarello.e/data/refer/refcocog_umd_gt_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/mscoco/images/train2014:/workspace/images:ro -v /data/bugliarello.e/data/refer/refcocog_umd_gt_imgfeat:/workspace/features -v /data/bugliarello.e/data/refer:/workspace/annotations:ro -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=5 python extract_refcocog_umd_gt_image.py --group_id 0 --total_group 1


### NLVR2
mkdir -p /data/bugliarello.e/data/nlvr/nlvr2/nlvr2_imgfeat
cp $(pwd)/data/nlvr2_imgfeat/extract_nlvr2_image.py /data/bugliarello.e/data/nlvr/nlvr2/nlvr2_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/nlvr/nlvr2/images:/workspace/images:ro -v /data/bugliarello.e/data/nlvr/nlvr2/nlvr2_imgfeat:/workspace/features -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=5 python extract_nlvr2_image.py --group_id 0 --total_group 5
CUDA_VISIBLE_DEVICES=6 python extract_nlvr2_image.py --group_id 1 --total_group 5


### V7W
mkdir -p /data/bugliarello.e/data/visual7w-toolkit/datasets/v7w_imgfeat
cp $(pwd)/data/v7w_imgfeat/extract_v7w_image.py /data/bugliarello.e/data/visual7w-toolkit/datasets/v7w_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/visual7w-toolkit/datasets/images:/workspace/images:ro -v /data/bugliarello.e/data/visual7w-toolkit/datasets/v7w_imgfeat:/workspace/features -v /data/bugliarello.e/data/visual7w-toolkit/datasets:/workspace/annotations:ro -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=6 python extract_v7w_image.py --group_id 0 --total_group 1

mkdir -p /data/bugliarello.e/data/visual7w-toolkit/datasets/v7w_gt_imgfeat
cp $(pwd)/data/v7w_imgfeat/extract_v7w_gt_image.py /data/bugliarello.e/data/visual7w-toolkit/datasets/v7w_gt_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/visual7w-toolkit/datasets/images:/workspace/images:ro -v /data/bugliarello.e/data/visual7w-toolkit/datasets/v7w_gt_imgfeat:/workspace/features -v /data/bugliarello.e/data/visual7w-toolkit/datasets:/workspace/annotations:ro -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
cd /workspace/features
CUDA_VISIBLE_DEVICES=7 python extract_v7w_gt_image.py --group_id 0 --total_group 1


### GuessWhat
mkdir -p /data/bugliarello.e/data/guesswhat/guesswhat_gt_imgfeat
cp $(pwd)/data/guesswhat_imgfeat/extract_guesswhat_gt_image.py /data/bugliarello.e/data/guesswhat/guesswhat_gt_imgfeat
docker run --gpus all -v /data/bugliarello.e/data/mscoco/images:/workspace/images:ro -v /data/bugliarello.e/data/guesswhat/guesswhat_gt_imgfeat:/workspace/features -v /data/bugliarello.e/data/guesswhat/annotations:/workspace/annotations:ro -v $(pwd)/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
pip install json-lines
cd /workspace/features
CUDA_VISIBLE_DEVICES=9 python extract_guesswhat_gt_image.py --group_id 0 --total_group 1


### CC
docker run --gpus all -v /tmp/ema/data/conceptual_captions:/workspace/images:ro -v /tmp/ema/data/cc_imgfeat:/workspace/features -v /tmp/ema/snap:/workspace/snap --rm -it airsplay/bottom-up-attention bash
docker run --gpus all -v /home/bugliarello.e/data/conceptual_captions/training.tar:/workspace/images:ro -v $(pwd)/data/cc_imgfeat:/workspace/features -v $(pwd)/snap:/workspace/snap:ro --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.0
CUDA_VISIBLE_DEVICES=5 python extract_cc_image.py --split train --group_id 0 --total_group 8

docker run --gpus all -v /home/bugliarello.e/data/conceptual_captions:/workspace/images:ro -v $(pwd)/data/cc_imgfeat:/workspace/features -v $(pwd)/snap:/workspace/snap:ro --rm -it airsplay/bottom-up-attention bash
pip install python-dateutil==2.5.1
CUDA_VISIBLE_DEVICES=0 python extract_cc_image.py --split valid --group_id 0 --total_group 4


## Pretrained models for VL-BERT

wget --header="Host: doc-10-a4-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,it;q=0.8,ja;q=0.7" --header="Referer: https://drive.google.com/" --header="Cookie: AUTH_f4k8g0tslhvcm6fmv2c5eu98j4d8ropj_nonce=8u0sppif4qbie" --header="Connection: keep-alive" "https://doc-10-a4-docs.googleusercontent.com/docs/securesc/cfnd6k8evm1fc2hgma7tsh3it4pm1ksf/tgco1tu52pncensmagevt6r4rps30vir/1591372800000/15420328229527158211/07893326812593803856/14VceZht89V5i54-_xWiw58Rosa5NDL2H?e=download&authuser=0&nonce=8u0sppif4qbie&user=07893326812593803856&hash=65tfhpmvf9qfuo3m3t2cgrra7hhphv32" -c -O 'bert.zip'

wget --header="Host: doc-08-a4-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,it;q=0.8,ja;q=0.7" --header="Referer: https://drive.google.com/" --header="Cookie: AUTH_f4k8g0tslhvcm6fmv2c5eu98j4d8ropj=07893326812593803856|1591372800000|mi0cqdg23vfju42p8c38g8rbf7idv6ai" --header="Connection: keep-alive" "https://doc-08-a4-docs.googleusercontent.com/docs/securesc/cfnd6k8evm1fc2hgma7tsh3it4pm1ksf/di2v11aef9sc77kv8clatd2f5nvscoe1/1591372950000/15420328229527158211/07893326812593803856/1qJYtsGw1SfAyvknDZeRBnp2cF4VNjiDE?e=download&authuser=0" -c -O 'resnet101-pt-vgbua-0000.model'
