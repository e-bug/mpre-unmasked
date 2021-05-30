# Multimodal Pretraining Unmasked

This is the implementation of the approaches described in the paper:
> Emanuele Bugliarello, Ryan Cotterell, Naoaki Okazaki and Desmond Elliott. [Multimodal Pretraining Unmasked: A Meta-Analysis and a Unified Framework of Vision-and-Language BERTs](https://arxiv.org/abs/2011.15124). Transactions of the Association for Computational Linguistics, 2021.

We provide the code for reproducing our results, as well as log files.
Preprocessed data and pretrained models are also available in [VOLTA](https://github.com/e-bug/volta).

**NB:** This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
During cluster maintenance, a small portion of data preparation and log files have been lost.
Nevertheless, this repository contains the core software to reproduce our results.
The missing data preparation files were derived from the official repositories of LXMERT, ViLBERT and VL-BERT, available under `code/`.


## Requirements

You can clone this repository issuing: <br>
`git clone git@github.com:e-bug/mpre-unmasked`

The Python environments for each code base (LXMERT, ViLBERT, VL-BERT, VOLTA) can be installed from the corresponding directories in `code/`.


## Data

Check out [`data/`](data) for download and preprocessing steps.
A clean, step-by-step version and preprocessed features are available in [VOLTA](https://github.com/e-bug/volta/blob/main/data/README.md).


## Models

Check out [`MODELS.md`](https://github.com/e-bug/volta/blob/main/MODELS.md) in VOLTA for links to pretrained models.


## Training and Evaluation

We provide our scripts to train (i.e. pretrain or fine-tune) and evaluate models in [experiments/](experiments).
These include ViLBERT, LXMERT and VL-BERT using the official repositories, 
as well as ViLBERT, LXMERT, VL-BERT, VisualBERT and UNITER using VOLTA.


## License

This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you find our code/data/models or ideas useful in your research, please consider citing the paper:
```
@article{bugliarello-etal-2021-multimodal,
    title = "Multimodal Pretraining Unmasked: {A} Meta-Analysis and a Unified Framework of Vision-and-Language {BERT}s",
    author = "Bugliarello, Emanuele and
      Cotterell, Ryan and
      Okazaki, Naoaki and
      Elliott, Desmond",
    journal = "Transactions of the Association for Computational Linguistics",
    year = "2021",
    url = "https://arxiv.org/abs/2011.15124",
}
```


## Acknowledgement

Our codebase heavily relies on these excellent repositories:
- [vilbert-multi-task](https://github.com/facebookresearch/vilbert-multi-task)
- [vilbert_beta](https://github.com/jiasenlu/vilbert_beta)
- [lxmert](https://github.com/airsplay/lxmert)
- [VL-BERT](https://github.com/jackroos/VL-BERT)
- [visualbert](https://github.com/uclanlp/visualbert)
- [UNITER](https://github.com/ChenRocks/UNITER)
- [pytorch-transformers](https://github.com/huggingface/pytorch-transformers)
- [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)
