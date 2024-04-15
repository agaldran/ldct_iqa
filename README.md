# Overview:
This repository contains the code for the first place in the competition on [Low-dose Computed Tomography Perceptual Image Quality Assessment](https://ldctiqac2023.grand-challenge.org/).
The data needed to train all models can be downloaded from [zenodo](https://zenodo.org/record/7833096#.ZEFywOxBzn5). The structure of this project is relatively straightforward, once the data is in place.
A jupyter notebook shows how I split the data into five train/validation sets.
As a result, 10 csv files are generated, e.g. tr_f1.csv, vl_f1.csv.
These can then be used for carrying out a training run, and we do five of them per model, one is a CNN (resnext50) and the other a transformer (swin), like this:
```
python train_PMH.py --csv_train data/train_f1.csv --model swin --n_heads 4 --overall_loss rps --hypar 10 --save_path swin/4hml_rps10_f1_104 --seed 1 --cycle_lens 10/4
python train_PMH.py --csv_train data/train_f2.csv --model swin --n_heads 4 --overall_loss rps --hypar 10 --save_path swin/4hml_rps10_f2_104 --seed 2 --cycle_lens 10/4
python train_PMH.py --csv_train data/train_f3.csv --model swin --n_heads 4 --overall_loss rps --hypar 10 --save_path swin/4hml_rps10_f3_104 --seed 3 --cycle_lens 10/4
python train_PMH.py --csv_train data/train_f4.csv --model swin --n_heads 4 --overall_loss rps --hypar 10 --save_path swin/4hml_rps10_f4_104 --seed 4 --cycle_lens 10/4
python train_PMH.py --csv_train data/train_f5.csv --model swin --n_heads 4 --overall_loss rps --hypar 10 --save_path swin/4hml_rps10_f5_104 --seed 5 --cycle_lens 10/4
```

Probably the most interesting bit here is that we use multi-head models, which we have shown to be nicely calibrated:

```
Multi-Head Multi-Loss Model Calibration 
Adrian Galdran, Johan Verjans, Gustavo Carneiro, and Miguel A. González Ballester
MICCAI 2024 [link](https://arxiv.org/abs/2303.01099)
```

Also, the loss function (which you can find in the `utils` folder) is inspired in the Ranked Probabilty Score idea, which we reviewed here:
```
Performance Metrics for Probabilistic Ordinal Classifiers
Adrian Galdran, MICCAI 2024 [link](https://arxiv.org/abs/2309.08701)
```


