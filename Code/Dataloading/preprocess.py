from collections import Counter
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def update_dict(dict, key, value=None):
    """
    Function for updating a dictionary.
    Inputs:
        dict - Dictionary to update
        key - Key to update
        value - Value to update the key with
    """
    
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            dict[key] = value

  
def read_news_data(news_path, args, mode='train'):
    """
    Function for reading the news.
    Inputs:
        news_path - Path to the news file
        args - Parsed arguments
        mode - Indicating whether to read news as train or test
    Outputs:
        news - Index of news articles
        news_index - Index for the news ids to index
        category_dict - Category dictionary
        subcategory_dict - Subcategory dictionary
        word_dict - Word dictionary
    """
    
    language_dict = {
      'MIND': 'english',
      'RTL_NR': 'dutch',
    }
    
    news = {}
    category_dict = {}
    subcategory_dict = {}
    news_index = {}
    word_cnt = Counter()
    
    # Loop over the files
    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            if args.dataset == 'MIND':          
              if len(splited) > 8:
                splited = splited[:8]
              doc_id, category, subcategory, title, abstract, _, _, _ = splited
            else:
              doc_id, category, title, abstract = splited
            update_dict(news_index, doc_id)
            
            # Update the dicts
            title = title.lower()
            title = word_tokenize(title, language=language_dict[args.dataset])
            abstract = abstract.lower()
            abstract = word_tokenize(abstract, language=language_dict[args.dataset])
            if args.dataset == 'MIND':
              update_dict(news, doc_id, [title, abstract, category, subcategory])
            else:
              update_dict(news, doc_id, [title, abstract, category])
            if mode == 'train':
              update_dict(category_dict, category)
              if args.dataset == 'MIND':
                update_dict(subcategory_dict, subcategory)
              word_cnt.update(title)
              word_cnt.update(abstract)
    
    # Create word dict
    if mode == 'train':
        word = [k for k, v in word_cnt.items() if v > args.filter_num]
        word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
        return news, news_index, category_dict, subcategory_dict, word_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'


def get_doc_input(news, news_index, category_dict, subcategory_dict, word_dict, args):
    """
    Function for getting the news article input.
    Inputs:
        news - Index of news articles
        news_index - Index for the news ids to index
        category_dict - Category dictionary
        subcategory_dict - Subcategory dictionary
        word_dict - Word dictionary
        args - Parsed arguments
    Outputs:
        news_title - News title token ids
        news_abstract - News abstract token ids
        news_category - News category representation
        news_subcategory - News subcategory representation
    """
    
    # Create the inputs
    news_num = len(news) + 1
    news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
    news_abstract = np.zeros((news_num, args.num_words_abstract), dtype='int32')
    news_category = np.zeros((news_num, 1), dtype='int32')
    news_subcategory = np.zeros((news_num, 1), dtype='int32')
    
    # Encode the features
    for key in tqdm(news):
        if args.dataset == 'MIND':
          title, abstract, category, subcategory = news[key]
        else:
          title, abstract, category = news[key]
        doc_index = news_index[key]

        for word_id in range(min(args.num_words_title, len(title))):
            if title[word_id] in word_dict:
                news_title[doc_index, word_id] = word_dict[title[word_id]]
        
        for word_id in range(min(args.num_words_abstract, len(abstract))):
            if abstract[word_id] in word_dict:
                news_abstract[doc_index, word_id] = word_dict[abstract[word_id]]

        news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        if args.dataset == 'MIND':
          news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0
    
    if args.dataset == 'MIND':
      return [news_title, news_abstract, news_category, news_subcategory]
    else:
      return [news_title, news_abstract, news_category]
  

def read_news_data_PLM(news_path, args, tokenizer, mode='train'):
    """
    Function for reading the news using a tokenizer.
    Inputs:
        news_path - Path to the news file
        args - Parsed arguments
        tokenizer - Tokenizer
        mode - Indicating whether to read news as train or test
    Outputs:
        news - Index of news articles
        news_index - Index for the news ids to index
        category_dict - Category dictionary
        subcategory_dict - Subcategory dictionary
    """
    
    news = {}
    categories = []
    subcategories = []
    news_index = {}
    index = 1
    
    # Loop over the files
    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            if args.dataset == 'MIND':          
              if len(splited) > 8:
                splited = splited[:8]
              doc_id, category, subcategory, title, abstract, _, _, _ = splited
            else:
              doc_id, category, title, abstract = splited
            news_index[doc_id] = index
            index += 1
            
            # Tokenize the title and abstract
            title = tokenizer(title, max_length=args.num_words_title, padding='max_length', truncation=True)
            abstract = tokenizer(abstract, max_length=args.num_words_abstract, padding='max_length', truncation=True)
            
            # Add the category and subcategory
            categories.append(category)
            if args.dataset == 'MIND':
              subcategories.append(subcategory)
            
            # Save the features
            if args.dataset == 'MIND':
              news[doc_id] = [title, abstract, category, subcategory]
            else:
              news[doc_id] = [title, abstract, category]
    
    # Create category and subcategory dictionaries if training
    if mode == 'train':
        categories = list(set(categories))
        category_dict = {}
        index = 1
        for x in categories:
            category_dict[x] = index
            index += 1
        
        if args.dataset == 'MIND':
          subcategories = list(set(subcategories))
          subcategory_dict = {}
          index = 1
          for x in subcategories:
            subcategory_dict[x] = index
            index += 1
        else:
          subcategory_dict = {}

        return news, news_index, category_dict, subcategory_dict, None
    elif mode == 'test':
        return news, news_index


def get_doc_input_PLM(news, news_index, category_dict, subcategory_dict, args):
    """
    Function for getting the news article input.
    Inputs:
        news - Index of news articles
        news_index - Index for the news ids to index
        category_dict - Category dictionary
        subcategory_dict - Subcategory dictionary
        args - Parsed arguments
    Outputs:
        news_title - News title token ids
        news_title_type - News title token type ids
        news_title_attmask - News title attention mask
        news_abstract - News abstract token ids
        news_abstract_type - News abstract token type ids
        news_abstract_attmask - News abstract attention mask
        news_category - News category representation
        news_subcategory - News subcategory representation
    """
    
    # Add the title
    news_num = len(news) + 1
    news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
    news_title_type = np.zeros((news_num, args.num_words_title), dtype='int32')
    news_title_attmask = np.zeros((news_num, args.num_words_title), dtype='int32')
    
    # Add the abstract
    news_abstract = np.zeros((news_num, args.num_words_abstract), dtype='int32')
    news_abstract_type = np.zeros((news_num, args.num_words_abstract), dtype='int32')
    news_abstract_attmask = np.zeros((news_num, args.num_words_abstract), dtype='int32')

    # Add the category and subcategory
    news_category = np.zeros((news_num, 1), dtype='int32')
    if args.dataset == 'MIND':
      news_subcategory = np.zeros((news_num, 1), dtype='int32')
    else:
      news_subcategory = None
    
    # Add the different news articles to the news representations
    for key in tqdm(news):
        if args.dataset == 'MIND':
          title, abstract, category, subcategory = news[key]
        else:
          title, abstract, category = news[key]
        doc_index = news_index[key]
        
        # Add the title
        news_title[doc_index] = title['input_ids']
        news_title_type[doc_index] = title['token_type_ids']
        news_title_attmask[doc_index] = title['attention_mask']          

        # Add the abstract
        news_abstract[doc_index] = abstract['input_ids']
        news_abstract_type[doc_index] = abstract['token_type_ids']
        news_abstract_attmask[doc_index] = abstract['attention_mask']

        # Add the category and subcategory
        news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        if args.dataset == 'MIND':
          news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0
    
    # Return the nes representations
    return [news_title, news_title_type, news_title_attmask, news_abstract, news_abstract_type,
            news_abstract_attmask, news_category, news_subcategory]

    
def read_news_data_sentencetransformer(news_path, args):
    """
    Function for reading the news using a sentence transformer.
    Inputs:
        news_path - Path to the news file
        args - Parsed arguments
    Outputs:
        news - Numpy array of news article embeddings
    """
    
    news = []
    news.append(np.zeros((1, 512), dtype='int32'))
    
    # Initialize the sentence transformer model
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    
    # Loop over the files
    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            if args.dataset == 'MIND':          
              if len(splited) > 8:
                splited = splited[:8]
              doc_id, category, subcategory, title, abstract, _, _, _ = splited
            else:
              doc_id, category, title, abstract = splited
            
            # Encode the combination of title and abstract
            if isinstance(abstract, str):
              content = title + abstract
            else:
              content = title
            embedding = model.encode(content)
            embedding = np.expand_dims(embedding, axis=0)
            
            # Save the embedding
            news.append(embedding)
        
        # Return the news embeddings
        news = np.squeeze(np.array(news))
        return news


def read_user_ids(behavior_path, args):
    """
    Function for reading the user ids from the behaviors.
    Inputs:
        behavior_path - Path to the news file
        args - Parsed arguments
    Outputs:
        user_dict - User dictionary
    """
    
    user_dict = {}
    
    # Loop over the files
    with tf.io.gfile.GFile(behavior_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            user_id = splited[2]
            update_dict(user_dict, user_id)
    
    # Return the user dictionary
    return user_dict