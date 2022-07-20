# PD-Rec

This repository contains research on diversity in personalized news recommendation. We research the effect of personalized levels of diversity in personalized news recommendation. We use a novel re-ranking algorithm, called Personalized Diverse Recommendation (PD-Rec), to introduce diversity to state-of-the-art news recommenders.

This work was done for the Master thesis for the master Artificial Intelligence @ UvA (University of Amsterdam)

## Content

This repository consists of the following main scripts and folders:

- **Hyperparameter Search**: this folder contains all the results from the hyperparameter search.
- **Experimental Results**: this folder contains all the results from the experiments, ordered per experiment.
- **Code**: this folder contains all the source code.
- **Code/main.py**: this python file is the main file to call when running the experiments.
- **Code/baselines.py**: this python file is the file to call when running the full diversity and random baselines.

The accompanying thesis of this research can be found in this repository as **Thesis.pdf**.

## Dataset

Our re-split version of the MIND dataset can be found in [this Drive folder](https://drive.google.com/file/d/1yuM6JZGUDCuBXzbBDV6IcQ1g2LBd6Mji/view?usp=sharing). Due to privacy laws and considerations we are unable to make the RTL-NR dataset public.

## Prerequisites

- Anaconda. Available at: https://www.anaconda.com/distribution/

## Getting Started

1. Open Anaconda prompt and clone this repository (or download and unpack zip):

```bash
git clone https://github.com/Luuk99/MasterThesis.git
```

2. Create the environment:

```bash
conda env create -f environments/environment.yml
```

3. Activate the environment:

```bash
conda activate PDRec
```

4. Move to the directory:

```bash
cd MasterThesis/Code
```

5. Download our MIND dataset split and unzip into a folder called MIND:

6. Run the training function with default parameters:

```bash
python main.py
```

## Arguments

The models can be trained and evaluated using the following command line arguments:

```bash
usage: main.py [-h] [--mode MODE] [--dataset DATASET] [--root_data_dir ROOT_DATA_DIR] [--train_dir TRAIN_DIR]
                    [--dev_dir DEV_DIR] [--test_dir TEST_DIR] [--model_dir MODEL_DIR] [--load_ckpt_name LOAD_CKPT_NAME]
                    [--save_dev_results SAVE_DEV_RESULTS] [--seed SEED] [--epochs EPOCHS] [--optimizer OPTIMIZER] [--lr LR]
                    [--batch_size BATCH_SIZE] [--npratio NPRATIO] [--enable_gpu ENABLE_GPU]
                    [--shuffle_buffer_size SHUFFLE_BUFFER_SIZE] [--num_workers NUM_WORKERS] [--log_steps LOG_STEPS]
                    [--patience PATIENCE] [--filter_num FILTER_NUM] [--architecture ARCHITECTURE]
                    [--num_unfrozen_layers NUM_UNFROZEN_LAYERS] [--num_words_title NUM_WORDS_TITLE]
                    [--num_words_abstract NUM_WORDS_ABSTRACT] [--user_log_length USER_LOG_LENGTH]
                    [--news_encoder_model NEWS_ENCODER_MODEL] [--news_dim NEWS_DIM]
                    [--news_query_vector_dim NEWS_QUERY_VECTOR_DIM] [--user_query_vector_dim USER_QUERY_VECTOR_DIM]
                    [--num_attention_heads NUM_ATTENTION_HEADS] [--drop_rate DROP_RATE] [--do_lower_case DO_LOWER_CASE]
                    [--diversify DIVERSIFY] [--similarity_measure SIMILARITY_MEASURE] [--reranking_function RERANKING_FUNCTION]
                    [--s_min S_MIN] [--s_max S_MAX] [--d_min D_MIN] [--d_max D_MAX] [--fixed_s FIXED_S]

optional arguments:
  -h, --help                                      Show help message and exit.
  --mode MODE                                     Whether to train, test or both. Options: ['train', 'test', 'train_test']. Default is 'train'.
  --dataset DATASET                               Which dataset to use. Options: ['MIND', 'RTL_NR']. Default is 'MIND'.
  --root_data_dir ROOT_DATA_DIR                   Root directory of the data. Default is './MIND'.
  --train_dir TRAIN_DIR                           Directory within the root_data_dir where the training data is stored. Default is 'train'.
  --dev_dir DEV_DIR                               Directory within the root_data_dir where the dev data is stored. Default is 'dev'.
  --test_dir TEST_DIR                             Directory within the root_data_dir where the test data is stored. Default is 'test'.
  --model_dir MODEL_DIR                           Directory where the model is saved and loaded from. Default is './model'.
  --load_ckpt_name LOAD_CKPT_NAME                 Checkpoint to load during testing. Default is None (don\'t load any).
  --save_dev_results SAVE_DEV_RESULTS             Boolean indicating whether to save the evaluation results (recommendations, user similarity and candidate diversity). Default is False.
  --seed SEED                                     Seed to use during training. Default is 1234.
  --epochs EPOCHS                                 Number of epochs to train for. Default is 10.
  --optimizer OPTIMIZER                           Optimizer to use for training. Options: ['adam', 'adamw', 'sgd', 'sgd_momentum']. Default is 'adam'.
  --lr LR                                         Learning rate to use during training. Default is 1e-4.
  --batch_size BATCH_SIZE                         Batch size to use during training and testing. Default is 64.
  --npratio NPRATIO                               Number of negatives to sample per positive training example. Default is 4.
  --enable_gpu ENABLE_GPU                         Boolean indicating whether to use the GPU or not. Default is True.
  --shuffle_buffer_size SHUFFLE_BUFFER_SIZE       Buffer size for shuffling the dataset. Default is 10000.
  --num_workers NUM_WORKERS                       Number of workers for data loading. Default is 0.
  --log_steps LOG_STEPS                           Number of steps before logging intermediate metrics. Default is 1000.
  --patience PATIENCE                             Number of epochs without improvement in AUC before stopping. Default is 3.
  --filter_num FILTER_NUM                         Number of word occurences to filter out. Default is 0 (no filtering).
  --architecture ARCHITECTURE                     Which model architecture to use. Options: ['plm4newsrec', 'nrms', 'naml', 'lstur']. Default is 'plm4newsrec'.
  --num_unfrozen_layers NUM_UNFROZEN_LAYERS       Number of last layers to keep unfrozen in the encoder model. Default is 2.
  --num_words_title NUM_WORDS_TITLE               Maximum token length of the title. Default is 20.
  --num_words_abstract NUM_WORDS_ABSTRACT         Maximum token length of the abstract. Default is 50.
  --user_log_length USER_LOG_LENGTH               Maximum length of the user history. Default is 50.
  --news_encoder_model NEWS_ENCODER_MODEL         Text encoder model for the news content (only used in plm4newsrec architecture). Options: ['bert', 'fastformer']. Default is 'fastformer'.
  --news_dim NEWS_DIM                             Dimensionality of the news embeddings. Default is 64.
  --news_query_vector_dim NEWS_QUERY_VECTOR_DIM   Dimensionality of the news query vector. Default is 200.
  --user_query_vector_dim USER_QUERY_VECTOR_DIM   Dimensionality of the user query vector. Default is 200.
  --num_attention_heads NUM_ATTENTION_HEADS       Number of attention heads. Default is 20.
  --drop_rate DROP_RATE                           Dropout rate to use during training. Default is 0.2.
  --do_lower_case DO_LOWER_CASE                   Boolean indicating whether to lower case the data when using NRMS or NAML. Default is True.
  --diversify DIVERSIFY                           Boolean indicating whether to diversify the recommendations. Default is False.
  --similarity_measure SIMILARITY_MEASURE         Similarity measure to use for diversification, only used when diversify=True. Options: ['cosine_similarity', 'euclidean_distance']. Default is 'cosine_similarity'.
  --reranking_function RERANKING_FUNCTION         Reranking function to use for diversification, only used when diversify=True. Options: ['naive', 'bound', 'normalized']. Default is 'naive'.
  --s_min S_MIN                                   Minimimum user similarity value, only used when reranking_function is either 'bound' or 'normalized'. Default is 0.0 (not used).
  --s_max S_MAX                                   Maximum user similarity value, only used when reranking_function is either 'bound' or 'normalized'. Default is 0.0 (not used).
  --d_min D_MIN                                   Minimimum candidate diversity value, only used when reranking_function is 'normalized'. Default is 0.0 (not used).
  --d_max D_MAX                                   Maximum candidate diversity value, only used when reranking_function is 'normalized'. Default is 0.0 (not used).
  --fixed_s FIXED_S                               Fixed user similarity weight. Default is 0.0 (weight is determined per user).
```

For the baselines, the same set of parameters apply but not all are used. The default parameters will suffice here.

## Authors

- Luuk Kaandorp - luuk.kaandorp@student.uva.nl

## Acknowledgements

- The re-split MIND dataset is taken from the original [MIND website](https://msnews.github.io/).
- Much of the original training loop is adapted from the orignal [PLM4NewsRec Github](https://github.com/wuch15/PLM4NewsRec).
- The NRMS and NAML models have been adapted from this [PyTorch news recommenders Github](https://github.com/yflyl613/NewsRecommendation).
