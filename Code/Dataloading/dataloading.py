import os
import numpy as np
import torch

from dataloader import DataloaderTrain, DataLoaderTest
from preprocess import read_news_data, read_news_data_PLM, get_doc_input, get_doc_input_PLM, read_news_data_sentencetransformer, read_user_ids


def load_data(data_dir, args, tokenizer, dataset='train', category_dict=None, subcategory_dict=None,
              model=None, word_dict=None, baseline=False):
    """
    Function for loading the data.
    Inputs:
        data_dir - Directory containing the data
        args - Parsed arguments
        tokenizer - Tokenizer instance for PLM4NewsRec
          None if not PLM4NewsRec architecture
        dataset - String indicating which dataset is used
        category_dict - Category dictionary
          None if loading training data
        subcategory_dict - Subcategory dictionary
          None if loading training data
        word_dict - Word dictionary
          None if loading training data
        model - Recommender model instance
          None if loading training data 
        baseline - Boolean indicating whether we are using random/diversity baselines
    Outputs:
        dataloader - Dataloader instance containing train or test data
        category_dict - Category dictionary
        subcategory_dict - Subcategory dictionary
        word_dict - Word dictionary
        user_id_dict - User id dictionary
    """
    
    # Select the right dataset and mode
    if dataset == 'train':
      data_dir = os.path.join(data_dir, args.train_dir)
      mode = 'train'
    elif dataset == 'dev':
      data_dir = os.path.join(data_dir, args.dev_dir)
      mode = 'test'
    elif dataset == 'test':
      data_dir = os.path.join(data_dir, args.test_dir)
      mode = 'test'
    
    # Read the news article data
    if tokenizer is not None:
      if mode == 'train':
        news, news_index, category_dict, subcategory_dict, word_dict = read_news_data_PLM(
          os.path.join(data_dir, 'news.tsv'), 
          args,
          tokenizer,
          mode=mode,
        )
      else:
        news, news_index = read_news_data_PLM(
          os.path.join(data_dir, 'news.tsv'), 
          args,
          tokenizer,
          mode=mode,
        )
    else:
      if mode == 'train':
        news, news_index, category_dict, subcategory_dict, word_dict = read_news_data(
          os.path.join(data_dir, 'news.tsv'), 
          args,
          mode=mode,
        )
      else:
        news, news_index = read_news_data(
          os.path.join(data_dir, 'news.tsv'), 
          args,
          mode=mode,
        )
    
    # Convert to tensors
    if tokenizer is not None:
      news_features = get_doc_input_PLM(news, news_index, category_dict, subcategory_dict, args)
    else:
      news_features = get_doc_input(news, news_index, category_dict, subcategory_dict, word_dict, args)
    
    # Concatenate the different representations
    news_combined = np.concatenate([
        x for x in
        news_features
        if x is not None], axis=1)
    
    # Read the user ids
    user_id_dict = read_user_ids(os.path.join(data_dir, 'behaviors.tsv'), args)
          
    if dataset == 'train':
      # Initialize the dataloader
      dataloader = DataLoaderTrain(
          news_index=news_index,
          news_combined=news_combined,
          word_dict=None,
          user_id_dict=user_id_dict,
          data_dir=data_dir,
          args=args,
          enable_prefetch=True,
          enable_shuffle=True,
      )
      
      # Return the dataloader, category, subcategory, word, and user id dictionaries
      return dataloader, category_dict, subcategory_dict, word_dict, user_id_dict
    else:
      if not baseline:
        # Convert the news articles to embeddings
        class NewsDataset(Dataset):
          def __init__(self, data):
            self.data = data

          def __getitem__(self, idx):
            return self.data[idx]

          def __len__(self):
            return self.data.shape[0]

        def news_collate_fn(arr):
          arr = torch.LongTensor(arr)
          return arr

        news_dataset = NewsDataset(news_combined)
        news_dataloader = DataLoader(news_dataset,
                                  batch_size=args.batch_size * 4,
                                  num_workers=args.num_workers,
                                  collate_fn=news_collate_fn)

        news_scoring = []
        with torch.no_grad():
          for input_ids in tqdm(news_dataloader):
            if args.enable_gpu:
              input_ids = input_ids.cuda()
            news_vec = model.news_encoder(input_ids)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_scoring.extend(news_vec)

        news_scoring = np.array(news_scoring)
      
      # Read the news articles using a sentence transformer
      st_embeddings = read_news_data_sentencetransformer(os.path.join(data_dir, 'news.tsv'), args)
      if baseline:
        news_scoring = st_embeddings
      
      # Initialize the dataloader
      dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring=news_scoring,
        st_embeddings=st_embeddings,
        word_dict=None,
        user_id_dict=user_id_dict,
        news_bias_scoring= None,
        data_dir=data_dir,
        args=args,
        enable_prefetch=True,
        enable_shuffle=False,
      )
      
      # Return the dataloader and number of articles
      return dataloader, np.shape(st_embeddings)[0] - 1