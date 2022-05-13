# MasterThesis

Master thesis for the master Artificial Intelligence @ UvA (University of Amsterdam)

This repository contains research on diversity in personalized news recommendation. We research the effect of diversity in personalized news recommendation. Both in an online and offline setting. We use a novel re-ranking algorithm to introduce diversity to state-of-the-art news recommenders.

## Content

This repository consists of the following scripts and folders:

- **?**: ?.

The accompanying thesis of this research can be found in this repository as **Thesis.pdf**.

## Dataset

Our re-split version of the MIND dataset can be found in [this Drive folder](https://drive.google.com/file/d/1yuM6JZGUDCuBXzbBDV6IcQ1g2LBd6Mji/view?usp=sharing). Due to privacy laws and considerations we are unable to make the RTL-NR dataset public.

## Prerequisites

- Anaconda. Available at: https://www.anaconda.com/distribution/

## Getting Started

1. Open Anaconda prompt and clone this repository (or download and unpack zip):

```bash
git clone https://github.com/AndrewHarrison/ir2
```

2. Create the environment:

```bash
conda env create -f environments/environment.yml
```

Or use the Lisa environment when running on the SurfSara Lisa cluster:

```bash
conda env create -f environments/environment_lisa.yml
```

3. Activate the environment:

```bash
conda activate ir2
```

4. Move to the directory:

```bash
cd ir2
```

5. Download the Natural Questions dataset:

```bash
python data_download/download_nq_data.py --resource data.retriever.nq --output_dir data/
```

6. Run the training script for DPR with a BERT encoder:

```bash
python train_dpr.py
```

Or download one of our models from the Drive folder and evaluate:

```bash
python evaluate_dpr.py
```

## Arguments

The DPR models can be trained using the following command line arguments:

```bash
usage: train_dpr.py [-h] [--model MODEL] [--max_seq_length MAX_SEQ_LENGTH] [--embeddings_size EMBEDDINGS_SIZE]
                        [--dont_embed_title] [--data_dir DATA_DIR] [--lr LR] [--warmup_steps WARMUP_STEPS]
                        [--dropout DROPOUT] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--save_dir SAVE_DIR]
                        [--seed SEED]

optional arguments:
  -h, --help                            Show help message and exit.
  --model MODEL                         What encoder model to use. Options: ['bert', 'distilbert', 'electra', 'tinybert']. Default is 'bert'.
  --max_seq_length MAX_SEQ_LENGTH       Maximum tokenized sequence length. Default is 256.
  --embeddings_size EMBEDDINGS_SIZE     Size of the model embeddings. Default is 0 (standard model embeddings sizes).
  --dont_embed_title                    Do not embed passage titles. Titles are embedded by default.
  --data_dir DATA_DIR                   Directory where the data is stored. Default is data/downloads/data/retriever/.
  --lr LR                               Learning rate to use during training. Default is 1e-5.
  --warmup_steps WARMUP_STEPS           Number of warmup steps. Default is 100.
  --dropout DROPOUT                     Dropout rate to use during training. Default is 0.1.
  --n_epochs N_EPOCHS                   Number of epochs to train for. Default is 40.
  --batch_size BATCH_SIZE               Training batch size. Default is 16.
  --save_dir SAVE_DIR                   Directory for saving the models. Default is saved_models/.
  --seed SEED                           Seed to use during training. Default is 1234.
```

The DPR models can be evaluated using the following command line arguments:

```bash
usage: evaluate_dpr.py [-h] [--model MODEL] [--load_dir LOAD_DIR] [--max_seq_length MAX_SEQ_LENGTH]
                        [--embeddings_size EMBEDDINGS_SIZE] [--batch_size BATCH_SIZE] [--dont_embed_title]
                        [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--seed SEED]

optional arguments:
  -h, --help                            Show help message and exit.
  --model MODEL                         What encoder model to use. Options: ['bert', 'distilbert', 'electra', 'tinybert']. Default is 'bert'.
  --load_dir LOAD_DIR                   Directory for loading the trained models. Default is saved_models/.
  --max_seq_length MAX_SEQ_LENGTH       Maximum tokenized sequence length. Default is 256.
  --embeddings_size EMBEDDINGS_SIZE     Size of the model embeddings. Default is 0 (standard model embeddings sizes).
  --batch_size BATCH_SIZE               Batch size to use for encoding questions and passages. Default is 512.
  --dont_embed_title                    Do not embed passage titles. Titles are embedded by default.
  --data_dir DATA_DIR                   Directory where the data is stored. Default is data/downloads/data/retriever/.
  --output_dir OUTPUT_DIR               Directory for saving the model evaluation metrics. Default is evaluation_outputs/.
  --seed SEED                           Seed to use during training. Default is 1234.
```

## Authors

- Luuk Kaandorp - luuk.kaandorp@student.uva.nl

## Acknowledgements

- The re-split MIND dataset is taken from the original [MIND website](https://msnews.github.io/).