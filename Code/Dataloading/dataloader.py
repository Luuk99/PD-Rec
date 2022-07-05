import sys
import traceback
import logging
import random
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import IterableDataset

from Dataloading.streaming import StreamSampler, StreamSamplerTest
import Utils.utils


def news_sample(news, ratio):
    """
    Function for sampling negative news.
    Inputs:
        news - List of negative news instances
        ratio - Number of samples to pick
    Outputs:
        sample - List of samples
    """
    
    if ratio > len(news):
        sample = news + [0] * (ratio - len(news))
    else:
        sample = random.sample(news, ratio)
    return sample

  
class DataLoaderTrain(IterableDataset):
    """
    Dataloader class for the training data.
    """
    
    def __init__(self, data_dir, args, news_index, news_combined, word_dict, user_id_dict, enable_prefetch=True, enable_shuffle=False):
        """
        Function for initializing the dataloader.
        Inputs:
            data_dir - Directory (folder) where the data is stored
            args - Parsed command line arguments
            news_index - Index for the combined news
            news_combined - Concatenated news features
            word_dict - Word dictionary
            user_id_dict - User id dictionary
            enable_prefetch - Enable the pre-fetching of batches
            enable_shuffle - Shuffle the data among workers
        """
        
        self.data_dir = data_dir

        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.batch_size = args.batch_size

        self.sampler = None

        self.shuffle_buffer_size = args.shuffle_buffer_size

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = args.enable_gpu
        self.epoch = -1

        self.news_combined = news_combined
        self.news_index = news_index
        self.word_dict = word_dict
        self.user_id_dict = user_id_dict

    def start(self):
        """
        Function for starting the iteration.
        """
        
        self.epoch += 1
        self.sampler = StreamSampler(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_buffer_size=self.shuffle_buffer_size,
            shuffle_seed=self.epoch,
        )
        self.sampler.__iter__()

    def trans_to_nindex(self, nids):
        """
        Function for converting news ids to index for look up.
        Inputs:
            nids - Ids of the news articles
        Outputs:
            index_ids - Ids of the articles within the index
        """
        
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        """
        Function for padding to a fixed length.
        Inputs:
            x - Item to be padded
            fix_length - Length to pad to
            padding_front - Boolean indicating padding at the front or back
            padding_value - Value to pad with
        Outputs:
            pad_x - Padded item
            mask - Mask for ignoring padding
        """
      
        if padding_front:
            pad_x = [padding_value] * (fix_length-len(x)) + x[-fix_length:]
            mask = [0] * (fix_length-len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[:fix_length] + [padding_value]*(fix_length-len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (len(x) - fix_length)
        return pad_x, mask

    def _produce(self):
        """
        Function for producing a batch.
        """
        
        try:
            self.epoch += 1
            self.sampler = StreamSampler(
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,
            )
            for batch in self.sampler:
                if self.stopped:
                    break
                context = self._process(batch)
                self.outputs.put(context)
                self.aval_count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def start_async(self):
        """
        Function for starting an asynchronous thread.
        """
        
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def parse_sent(self, sent, fix_length):
        """
        Function for parsing a sentence using the word dict.
        Inputs:
            sent - Sentence to parse
            fix_length - Length to pad the sentence to
        Outputs:
            sent - Parsed and padded sentence
        """
        
        sent = [self.word_dict[w] if w in self.word_dict else 0 for w in utils.word_tokenize(sent)]
        sent, _ = self.pad_to_fix_len(sent, fix_length, padding_front=False)
        return sent

    def parse_sents(self, sents, max_sents_num, max_sent_length, padding_front=True):
        """
        Function for parsing multiple sentences using the word dict.
        Inputs:
            sents - Sentences to parse
            max_sents_num - Maximum allowed sentences 
            max_sent_length - Length to pad the sentence to
            padding_front - Boolean indicating padding at the front or back
        Outputs:
            sents - Padded and parsed sentences
            sents_mask - Mask for ignoring padding
        """
      
        sents, sents_mask = self.pad_to_fix_len(sents, max_sents_num, padding_value='')
        sents = [self.parse_sent(s, max_sent_length) for s in sents]
        sents = np.stack(sents, axis=0)
        sents_mask = np.array(sents_mask)
        return sents, sents_mask

    def _process(self, batch):
        """
        Function for processing a batch.
        Inputs:
            batch - Batch of instances to process
        Outputs:
            user_feature_batch - Batch of the user history logs
            log_mask_batch - Batch of masks for the history logs
            news_feature_batch - Batch of canidate news articles
            label_batch - Batch of labels
            user_id_batch - Batch of user ids
        """
        
        batch_size = len(batch)
        batch_poss, batch = batch
        batch_poss = [x.numpy().decode("utf-8") for x in batch_poss]
        batch = [x.numpy().decode("utf-8").split("\t") for x in batch]
        label = 0
        user_feature_batch, log_mask_batch, news_feature_batch, label_batch, user_id_batch = [], [], [], [], []
        
        # Iterate over all lines in the batch
        for poss, line in zip(batch_poss, batch):
            click_docs = line[3].split()

            click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs),
                                             self.user_log_length)

            user_feature = self.news_combined[click_docs]

            sess_news = [i.split('-') for i in line[4].split()]
            sess_neg = [i[0] for i in sess_news if i[-1] == '0']
        
            poss = self.trans_to_nindex([poss])
            sess_neg = self.trans_to_nindex(sess_neg)

            if len(sess_neg) > 0:
                neg_index = news_sample(list(range(len(sess_neg))),
                                        self.npratio)
                sam_negs = [sess_neg[i] for i in neg_index]
            else:
                sam_negs = [0] * self.npratio
            sample_news = poss + sam_negs
            
            user_id = line[2]
            if user_id in self.user_id_dict:
              user_id_batch.append(self.user_id_dict[user_id])
            else:
              user_id_batch.append(0)

            news_feature = self.news_combined[sample_news]
            user_feature_batch.append(user_feature)
            log_mask_batch.append(log_mask)
            news_feature_batch.append(news_feature)
            label_batch.append(label)
        
        user_feature_batch = np.array(user_feature_batch)
        log_mask_batch = np.array(log_mask_batch)
        news_feature_batch = np.array(news_feature_batch)
        label_batch = np.array(label_batch)
        user_id_batch = np.array(user_id_batch)
        if self.enable_gpu:
            user_feature_batch = torch.LongTensor(user_feature_batch).cuda()
            log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()
            news_feature_batch = torch.LongTensor(news_feature_batch).cuda()
            label_batch = torch.LongTensor(label_batch).cuda()
            user_id_batch = torch.FloatTensor(user_id_batch).cuda()
        else:
            user_feature_batch = torch.LongTensor(user_feature_batch)
            log_mask_batch = torch.FloatTensor(log_mask_batch)
            news_feature_batch = torch.LongTensor(news_feature_batch)
            label_batch = torch.LongTensor(label_batch)
            user_id_batch = torch.FloatTensor(user_id_batch)

        return user_feature_batch, log_mask_batch, news_feature_batch, label_batch, user_id_batch
      
    def __iter__(self):
        """
        Necessary function for allowing iteration.
        """
        
        print("DataLoader __iter__()")
        if self.enable_prefetch:
            self.join()
            self.start_async()
        else:
            self.start()
        return self

    def __next__(self):
        """
        Function for accessing the next batch.
        Outputs:
            next_batch - Next batch from the sampler
        """
        
        if self.sampler and self.sampler.reach_end() and self.aval_count == 0:
            raise StopIteration
        if self.enable_prefetch:
            next_batch = self.outputs.get()
            self.outputs.task_done()
            self.aval_count -= 1
        else:
            next_batch = self._process(self.sampler.__next__())
        return next_batch

    def join(self):
        """
        Function for closing and joining the threads.
        """
        
        self.stopped = True
        if self.sampler:
            if self.enable_prefetch:
                while self.outputs.qsize() > 0:
                    self.outputs.get()
                    self.outputs.task_done()
                self.outputs.join()
                self.pool.shutdown(wait=True)
                logging.info("shut down pool.")
            self.sampler = None


class DataLoaderTest(IterableDataset):
    """
    Dataloader class for the test data.
    """
  
    def __init__(self, data_dir, args, news_index, news_scoring, st_embeddings, word_dict, user_id_dict, news_bias_scoring=None,
                 enable_prefetch=True, enable_shuffle=False):
        """
        Function for initializing the dataloader.
        Inputs:
            data_dir - Directory (folder) where the data is stored
            args - Parsed command line arguments
            news_index - Index for the combined news
            news_scoring - Embeddings of the news articles
            st_embeddings - Sentence transformer embeddings of the news articles
            word_dict - Word dictionary
            user_id_dict - User id dictionary
            news_bias_scoring - Optional bias for certain news articles
            enable_prefetch - Enable the pre-fetching of batches
            enable_shuffle - Shuffle the data among workers
        """
        
        self.data_dir = data_dir

        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.batch_size = args.batch_size
        
        self.sampler = None

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = args.enable_gpu
        self.epoch = -1

        self.news_scoring = news_scoring
        self.st_embeddings = st_embeddings
        self.news_bias_scoring = news_bias_scoring
        self.news_index = news_index
        self.word_dict = word_dict
        self.user_id_dict = user_id_dict

    def start(self):
        """
        Function for starting the iteration.
        """
        
        self.epoch += 1
        self.sampler = StreamSamplerTest(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_seed=self.epoch,
        )
        self.sampler.__iter__()
    
    def trans_to_nindex(self, nids):
        """
        Function for converting news ids to index for look up.
        Inputs:
            nids - Ids of the news articles
        Outputs:
            index_ids - Ids of the articles within the index
        """
        
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        """
        Function for padding to a fixed length.
        Inputs:
            x - Item to be padded
            fix_length - Length to pad to
            padding_front - Boolean indicating padding at the front or back
            padding_value - Value to pad with
        Outputs:
            pad_x - Padded item
            mask - Mask for ignoring padding
        """
        
        if padding_front:
            pad_x = [padding_value] * (fix_length-len(x)) + x[-fix_length:]
            mask = [0] * (fix_length-len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[:fix_length] + [padding_value]*(fix_length-len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (len(x) - fix_length)
        return pad_x, mask
    
    def _produce(self):
        """
        Function for producing a batch.
        """
        
        try:
            self.epoch += 1
            self.sampler = StreamSamplerTest(
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,
            )
            for batch in self.sampler:
                if self.stopped:
                    break
                context = self._process(batch)
                self.outputs.put(context)
                self.aval_count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise
    
    def start_async(self):
        """
        Function for starting an asynchronous thread.
        """
        
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def parse_sent(self, sent, fix_length):
        """
        Function for parsing a sentence using the word dict.
        Inputs:
            sent - Sentence to parse
            fix_length - Length to pad the sentence to
        Outputs:
            sent - Parsed and padded sentence
        """
        
        sent = [self.word_dict[w] if w in self.word_dict else 0 for w in utils.word_tokenize(sent)]
        sent, _ = self.pad_to_fix_len(sent, fix_length, padding_front=False)
        return sent

    def parse_sents(self, sents, max_sents_num, max_sent_length, padding_front=True):
        """
        Function for parsing multiple sentences using the word dict.
        Inputs:
            sents - Sentences to parse
            max_sents_num - Maximum allowed sentences 
            max_sent_length - Length to pad the sentence to
            padding_front - Boolean indicating padding at the front or back
        Outputs:
            sents - Padded and parsed sentences
            sents_mask - Mask for ignoring padding
        """
        
        sents, sents_mask = self.pad_to_fix_len(sents, max_sents_num, padding_value='')
        sents = [self.parse_sent(s, max_sent_length) for s in sents]
        sents = np.stack(sents, axis=0)
        sents_mask = np.array(sents_mask)
        return sents, sents_mask
    
    def _process(self, batch):
        """
        Function for processing a batch.
        Inputs:
            batch - Batch of instances to process
        Outputs:
            user_feature_batch - Batch of the user history logs
            log_mask_batch - Batch of masks for the history logs
            news_feature_batch - Batch of canidate news articles
            news_bias_batch - Batch of news biases
            label_batch - Batch of labels
            id_batch - Batch of candidate news ids
            entry_id_batch - Batch of indices of the test set
            user_st_embedding_batch - Batch of external embeddings in the user history
            candidate_st_embedding_batch - Batch of external embeddings of the candidate articles
            user_id_batch - Batch of user ids
        """
        
        batch_size = len(batch)
        _, batch = batch
        batch = [x.numpy().decode("utf-8").split("\t") for x in batch]

        user_feature_batch = [] 
        log_mask_batch = []
        news_feature_batch = []
        news_bias_batch = []
        label_batch = []
        id_batch = []
        entry_id_batch = []
        user_id_batch = []
        user_st_embedding_batch = []
        candidate_st_embedding_batch = []
        
        # Iterate over the lines in the batch
        for line in batch:
            entry_id_batch.append(int(line[0]))
            
            click_docs = line[3].split()

            click_docs, log_mask  = self.pad_to_fix_len(self.trans_to_nindex(click_docs),
                                             self.user_log_length)
            user_feature = self.news_scoring[click_docs]
            
            sample_news = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
            labels = [int(i.split('-')[1]) for i in line[4].split()]

            news_feature = self.news_scoring[sample_news]
            if self.news_bias_scoring is not None:
                news_bias = self.news_bias_scoring[sample_news]
            else:
                news_bias = [0] * len(sample_news)
            
            click_st_embedding = self.st_embeddings[click_docs]
            user_st_embedding_batch.append(click_st_embedding)
            
            candidate_st_embedding = self.st_embeddings[sample_news]
            candidate_st_embedding_batch.append(candidate_st_embedding)
            
            user_id = line[2]
            if user_id in self.user_id_dict:
              user_id_batch.append(self.user_id_dict[user_id])
            else:
              user_id_batch.append(0)
                
            user_feature_batch.append(user_feature)
            log_mask_batch.append(log_mask)
            news_feature_batch.append(news_feature)
            news_bias_batch.append(news_bias)
            label_batch.append(np.array(labels))
            id_batch.append([i.split('-')[0] for i in line[4].split()])
            
        user_feature_batch = np.array(user_feature_batch)
        log_mask_batch = np.array(log_mask_batch)
        label_batch = np.array(label_batch)
        user_id_batch = np.array(user_id_batch)
        if self.enable_gpu:
            user_feature_batch = torch.FloatTensor(user_feature_batch).cuda()
            log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()
            user_id_batch = torch.FloatTensor(user_id_batch).cuda()
        else:
            user_feature_batch = torch.FloatTensor(user_feature_batch)
            log_mask_batch = torch.FloatTensor(log_mask_batch)
            user_id_batch = torch.FloatTensor(user_id_batch)

        return user_feature_batch, log_mask_batch, news_feature_batch, news_bias_batch, label_batch, id_batch, entry_id_batch, user_st_embedding_batch, candidate_st_embedding_batch, user_id_batch
    
    def __iter__(self):
        """
        Necessary function for allowing iteration.
        """
        
        print("DataLoader __iter__()")
        if self.enable_prefetch:
            self.join()
            self.start_async()
        else:
            self.start()
        return self

    def __next__(self):
        """
        Function for accessing the next batch.
        Outputs:
            next_batch - Next batch from the sampler
        """
        
        if self.sampler and self.sampler.reach_end() and self.aval_count == 0:
            raise StopIteration
        if self.enable_prefetch:
            next_batch = self.outputs.get()
            self.outputs.task_done()
            self.aval_count -= 1
        else:
            next_batch = self._process(self.sampler.__next__())
        return next_batch

    def join(self):
        """
        Function for closing and joining the threads.
        """
        
        self.stopped = True
        if self.sampler:
            if self.enable_prefetch:
                while self.outputs.qsize() > 0:
                    self.outputs.get()
                    self.outputs.task_done()
                self.outputs.join()
                self.pool.shutdown(wait=True)
                logging.info("shut down pool.")
            self.sampler = None