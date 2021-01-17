# EffiCare

ICU predictions on MIMIC-III with discrete and distributed event representations.

## Updated results

We noticed an unwanted effect of average pooling, which leaks information on remaining number of steps. This informs the model on the decompensation and length-of-stay tasks. Removing this effect lowers the prediction performance on these tasks substantially, which is presented in the table below. Our latest work using GRU improves upon these results and achieves SOTA in [oguzserbetci/discrete-ehr](https://github.com/oguzserbetci/discrete-ehr).

|                                            | ('In-hospital mortality', 'AUC of PRC') | ('In-hospital mortality', 'AUC of ROC') | ('Decompensation', 'AUC of PRC') | ('Decompensation', 'AUC of ROC') | ('Length-of-stay', 'Kappa') | ('Length-of-stay', 'MAD')  | ('Phenotyping', 'Macro ROC AUC') | ('Phenotyping', 'Micro ROC AUC') | ('Phenotyping', 'Macro AUPRC') | ('Phenotyping', 'Micro AUPRC') |
| :----------------------------------------- | :-------------------------------------- | :-------------------------------------- | :------------------------------- | :------------------------------- | :-------------------------- | :------------------------- | :------------------------------- | :------------------------------- | :----------------------------- | :----------------------------- |
| Random embeddings (MT)                     | .559 (.503, .613)                       | .889 (.872, .905)                       | .278 (.268, .287)                | .905 (.903, .908)                | .264 (.262, .265)           | 103.911 (103.467, 104.35)  | .82 (.816, .823)                 | .851 (.848, .854)                | .52 (.512, .527)               | .576 (.569, .582)              |
| FastText embeddings (MT)                   | .606 (.553, .656)                       | .903 (.887, .917)                       | .316 (.306, .327)                | .922 (.92, .925)                 | .296 (.295, .298)           | 101.091 (100.611, 101.553) | .805 (.801, .809)                | .845 (.842, .848)                | .489 (.48, .496)               | .546 (.538, .553)              |
| with \*Reth19 embeddings (MT)              | .579 (.523, .634)                       | .895 (.879, .911)                       | .235 (.227, .244)                | .907 (.904, .91)                 | .286 (.285, .287)           | 101.925 (101.438, 102.433) | .819 (.816, .823)                | .857 (.855, .86)                 | .519 (.511, .526)              | .582 (.575, .589)              |
| FastText embeddings w/o event density (MT) | .607 (.553, .657)                       | .9 (.884, .915)                         | .267 (.258, .276)                | .922 (.92, .924)                 | .298 (.297, .3)             | 101.033 (100.545, 101.521) | .818 (.814, .822)                | .856 (.853, .858)                | .516 (.509, .524)              | .576 (.569, .583)              |
| FastText event count > 100 (MT)            | .605 (.552, .656)                       | .9 (.883, .915)                         | .265 (.257, .274)                | .926 (.924, .928)                | .301 (.299, .302)           | 101.039 (100.592, 101.491) | .818 (.814, .822)                | .857 (.854, .859)                | .513 (.506, .521)              | .574 (.567, .581)              |
| FastText only 17 \*Haru17/19 (MT)          | .495 (.44, .549)                        | .857 (.837, .876)                       | .188 (.181, .195)                | .889 (.886, .892)                | .289 (.288, .291)           | 101.906 (101.445, 102.367) | .773 (.768, .777)                | .819 (.816, .823)                | .421 (.413, .428)              | .497 (.489, .504)              |
| FastText w/o demographics (MT)             | .584 (.532, .634)                       | .898 (.882, .912)                       | .3 (.29, .309)                   | .919 (.916, .921)                | .303 (.302, .305)           | 100.807 (100.364, 101.258) | .808 (.804, .812)                | .842 (.839, .845)                | .491 (.483, .498)              | .54 (.533, .547)               |

## How-to generate input data

> You need [MIMIC-III](https://mimic.physionet.org/) access to run the code.

1. Setup the benchmark and FastText submodule:

```sh
git submodule update
```

2. Generate csv files for the benchmark tasks as explained in `mimic3-benchmarks/README.md` under _Building a benchmark_.
   You need to generate for all tasks not only multitask as the evaluation scripts in the benchmark depend on label files generated for each task.

3. Install FastText:

```sh
cd fastText
make
```

## How-to train on MIMIC-III

1. Setup a virtual environment with conda for easy CUDA support and install required packages using poetry:

```sh
conda create -n efficare pytorch==1.4.0 torchtext==0.4.0 tensorboard==2.0.0 python==3.7.5 -c pytorch
conda activate efficare
pip install poetry
poetry install
```

2. Extract bin-edges and patient sentences:

```sh
python -m dataloader.extractors
python -m dataloader.generate_demographics
```

This generates following files:

```sh
med_values.<table>*.txt
med_bin_edges.<table>*.txt
dem.*.params
embeddings/sentences.mimic3.txt.counts
embeddings/sentences.mimic3.txt
```

3. Train fasttext embeddings. You can skip this step as pretrained vectors are included in `embeddings/`:

```sh
./fastText/fasttext skipgram -ws 15 -minCount 1 -input embeddings/sentences.mimic3.txt -output embeddings/sentences.mimic3.txt.100d.Fasttext.15ws
```

4. Finetune on benchmark tasks with a multitask model as explained in the paper:

```sh
python -m finetune base -e 20
```

## Evaluation:

First we generate predictions as csv, and then use the evaluation scripts provided by Harutyunyan et al.

```sh
python -m collect <wandb_id>
sh evaluate.sh <wandb_id>
```

This results in files for predictions and the evaluation result for each task:

```sh
wandb/run-*-<wandb_id>/test_listfile_predictions/<task>-*.csv
wandb/run-*-<wandb_id>/test_listfile_predictions/<task>-*.csv.json
```
