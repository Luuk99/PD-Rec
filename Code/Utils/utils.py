# import logging
# import os
# import sys
# import torch
# import numpy as np
# import argparse
# import re

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfOutput, BertIntermediate, BertOutput


def initialize_tokenizer(args):
    """
    Function for initializing the tokenizer.
    Inputs:
        args - Parsed arguments
    Outputs:
        tokenizer - Initialized tokenizer
    """
    
    model_dict = { 
      'MIND': 'bert-base-uncased',
      'RTL_NR': 'GroNLP/bert-base-dutch-cased',
    }
    
    # Load the tokenizer
    pretrained_location = model_dict[args.dataset]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_location)
    
    # Return the tokenizer
    return tokenizer


def initialize_model(args, word_dict, category_dict, subcategory_dict):
    """
    Function for initializing the correct model.
    Inputs:
        args - Parsed arguments
        word_dict - Dictionary of words contained in the news articles
        category_dict - Dictionary of article categories
        subcategory_dict - Dictionary of article subcategories
    Outputs:
        model - Initialized model
    """
    
    model_dict = { 
      'MIND': ('bert-base-uncased', 768),
      'RTL_NR': ('GroNLP/bert-base-dutch-cased', 768),
    }
    
    # Check which architecture to use
    if args.architecture == 'nrms':
      # Load the glove embeddings
      matrix_dir = download_matrix(args)
      embedding_matrix = load_matrix(matrix_dir, word_dict, 300)
      # Load the nrms model
      model = NRMSModel(args, embedding_matrix)
    elif args.architecture == 'naml':
      # Load the glove embeddings
      matrix_dir = download_matrix(args)
      embedding_matrix = load_matrix(matrix_dir, word_dict, 300)
      # Load the naml model
      model = NAMLModel(args, embedding_matrix, len(category_dict), len(subcategory_dict))
    elif args.architecture == 'plm4newsrec':
      # Load the encoder model
      if args.news_encoder_model == 'fastformer':
        # Load the fastformer model
        config = BertConfig.from_json_file('/dbfs/mnt/rtl-databricks-datascience/lkaandorp/fastformer.json')
        matrix_dir = download_matrix(args)
        embedding_matrix = load_matrix(matrix_dir, word_dict, 300)
        news_encoder = FastformerModel(config, word_dict, embedding_matrix)
        token_embedding_dim = 256
      else:
        # Load the BERT model
        pretrained_location, token_embedding_dim = model_dict[args.dataset]
        news_encoder = AutoModel.from_pretrained(pretrained_location)
        
        # Freeze the layers of the model
        news_encoder = freeze_layers(news_encoder)
      
      # Load the recommender model
      model = PLM4NewsRecModel(args, news_encoder, token_embedding_dim, len(category_dict),
                               len(subcategory_dict))
    
    # Return the model
    return model


def freeze_layers(model, num_last_layers=2):
    """
    Function for freezing the final layers of the model.
    Inputs:
        model - Model to freeze the layers of
        num_last_layers - Number of last layers to leave unfrozen
    Outputs:
        model - Model with frozen layers
    """
    
    # Freeze all the layers of the model except the k last ones
    num_layers = model.config.num_hidden_layers
    last_layers = [str(num_layers - i) for i in range(1, num_last_layers + 1)]
    for name, param in model.named_parameters():
      if len(name.split('.')) > 3:
        if name.split('.')[2] in last_layers:
          param.requires_grad = True
        else:
          param.requires_grad = False
    
    # Return the model
    return model


def str2bool(v):
    """
    Function for parsing strings to boolean values.
    Inputs:
        v - String boolean value
    Outputs:
        bool - True if the string is a boolean and False otherwise
    """
    
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

        
def dump_args(args):
    """
    Function for printing all the parsed arguments.
    Inputs:
        args - Parsed arguments
    """
    
    for arg in dir(args):
        if not arg.startswith("_"):
            print(f"args[{arg}] = {getattr(args, arg)}")
            

def load_matrix(embedding_file_path, word_dict, word_embedding_dim):
    """
    Function for loading the GloVe embeddings matrix.
    Inputs:
        embedding_file_path - Path to the GloVe embeddings
        word_dict - Dictionary of words contained in the articles
        word_embedding_dim - Dimensionality of the word embeddings
    Outputs:
        embedding_matrix - Matrix containing the GloVe embeddings
    """
  
    embedding_matrix = np.zeros(shape=(len(word_dict) + 1, word_embedding_dim))
    if embedding_file_path is not None:
        with open(embedding_file_path, 'rb') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                word = line[0].decode()
                if word in word_dict:
                    index = word_dict[word]
                    tp = [float(x) for x in line[1:]]
                    embedding_matrix[index] = np.array(tp)
    return embedding_matrix
  

def get_checkpoint(directory, ckpt_name):
    """
    Function for loading a model checkpoint.
    Inputs:
        directory - Directory where the checkpoint is stored
        ckpt_name - Checkpoint name
    Outputs:
        ckpt_path - Path to the checkpoint
    """
    
    ckpt_path = os.path.join(directory, ckpt_name)
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        return None
      

def download_url(url, temp_dir):
    """
    Function for downloading a URL to a temporary file.
    Inputs:
        url - URL to download from
        temp_dir - Temporary directory
    Outputs:
        destination_filename - Filename and destination of the downloaded file
    """
    
    # Create a filename
    url_as_filename = url.replace('://', '_').replace('/', '_')
    destination_filename = os.path.join(temp_dir,url_as_filename)
    
    # Check if the file is already downloaded
    if os.path.isfile(destination_filename):
      print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url)))
      return destination_filename
    
    # Download the file otherwise
    print('Downloading file {} to {}'.format(os.path.basename(url), destination_filename), end='')
    urllib.request.urlretrieve(url, destination_filename, None)
    assert (os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    print('...done, {} bytes.'.format(nBytes))
    
    # Return the destination filename
    return destination_filename


def download_matrix(args):
    """
    Function for downloading an embedding matrix.
    Inputs:
        args - Parsed arguments
    Outputs:
        matrix_dir - Temporary directory containing the embeddings
    """
    
    url_dict = {
      'MIND': ('https://nlp.stanford.edu/data/glove.840B.300d.zip', 'glove.840B.300d.txt'),
      'RTL_NR': ('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz', 'embeddings.vec'),
    }
    
    # Make a temporary directory
    temp_dir = os.path.join(tempfile.gettempdir(), 'glove_embeddings')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Check if the embedings are already downloaded
    embedding_url, embedding_name = url_dict[args.dataset]
    if os.path.isfile(os.path.join(temp_dir, embedding_name)):
      matrix_dir = os.path.join(temp_dir, embedding_name)
      print('Bypassing download of already-downloaded embedding matrix {}'.format(matrix_dir))
      return matrix_dir
    
    # Download the data
    zip_path = download_url(embedding_url, temp_dir)
    if args.dataset == 'MIND':
      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    else:
      with gzip.open(zip_path, 'rb') as f_in:
        with open(os.path.join(temp_dir, 'embeddings.vec'), 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)
    
    # Return the embedding directory
    matrix_dir = os.path.join(temp_dir, embedding_name)
    return matrix_dir