# EffiCare

ICU predictions on MIMIC-III with discrete and distributed event representations.


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
``` sh
conda create -n efficare pytorch==1.4.0 torchtext==0.4.0 tensorboard==2.0.0 python==3.7.5 -c pytorch
conda activate efficare
pip install poetry
poetry install
```

2. Extract bin-edges and patient sentences:
``` sh
python -m dataloader.extractors
python -m dataloader.generate_demographics
```

This generates following files:
``` sh
med_values.<table>*.txt
med_bin_edges.<table>*.txt
dem.*.params
embeddings/sentences.mimic3.txt.counts
embeddings/sentences.mimic3.txt
```

3. Train fasttext embeddings. You can skip this step as pretrained vectors are included in `embeddings/`:
``` sh
./fastText/fasttext skipgram -ws 15 -minCount 1 -input embeddings/sentences.mimic3.txt -output embeddings/sentences.mimic3.txt.100d.Fasttext.15ws
```

4. Finetune on benchmark tasks with a multitask model as explained in the paper:
```sh
python -m finetune base -e 20
```


## Evaluation:

First we generate predictions as csv, and then use the evaluation scripts provided by Harutyunyan et al.
``` sh
python -m collect <wandb_id>
sh evaluate.sh <wandb_id>
```

This results in files for predictions and the evaluation result for each task:
```sh
wandb/run-*-<wandb_id>/test_listfile_predictions/<task>-*.csv
wandb/run-*-<wandb_id>/test_listfile_predictions/<task>-*.csv.json
```
