# import os
# import logging
# import fnmatch
# import random
# import numpy as np
# import tensorflow as tf

# import utils


class StreamReader:
    """
    StreamReader class for streaming the training data.
    """
    
    @tf.autograph.experimental.do_not_convert
    def __init__(self, data_paths, batch_size, shuffle=False, shuffle_buffer_size=1000):
        """
        Function for initializing the StreamReader.
        Inputs:
            data_paths - Paths to the data files
            batch_size - Size of the batches
            shuffle - Whether to shuffle the dataset
            shuffle_buffer_size - Size of the buffer for shuffling
        """
        
        tf.config.experimental.set_visible_devices([], device_type="GPU")
        path_len = len(data_paths)
        dataset = tf.data.Dataset.list_files(data_paths).interleave(
            lambda x: tf.data.TextLineDataset(x),
            cycle_length=path_len,
            block_length=128,
            num_parallel_calls=min(path_len, 64),
        )
        dataset = dataset.interleave(
            lambda x: tf.data.Dataset.from_tensor_slices(
                self._process_record(x)),
            cycle_length=path_len,
            block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        self.dataset = iter(dataset)
        self.session = None

    def _process_record(self, record):
        """
        Function for processing a record in the dataset.
        Inputs:
            record - Line from the dataset
        Outputs:
            sess_poss - Possitive impression
            repeated_record - Record repeated for the number of positive impressions
        """
        
        # iid, uid, time, his, impr
        records = tf.strings.split([record], '\t').values
        sess = tf.strings.split([records[4]], ' ').values  # (num)
        sess_label = tf.strings.split(sess, '-').values

        sess_poss = tf.gather(sess_label, tf.where(tf.equal(sess_label, '1'))-1)
        record = tf.expand_dims(record, axis=0)
        poss_num = tf.size(sess_poss)

        return sess_poss[:, 0], tf.repeat(record, poss_num, axis=0)

    def reset(self):
        """
        Function for resetting the stream.
        """
        
        self.endofstream = False

    def get_next(self):
        """
        Function for retrieving the next batch.
        Outputs:
            ret - Next batch in the dataset
              None if end of the dataset
        """
        
        try:
            ret = self.dataset.get_next()
        except tf.errors.OutOfRangeError:
            self.endofstream = True
            return None
        return ret

    def reach_end(self):
        """
        Function for assessing whether the end of the dataset has been reached.
        Outputs:
            self.endofstream - Boolean indicating the end of the stream 
        """
        
        return self.endofstream


class StreamSampler:
    """
    StreamSampler class for streaming the training data.
    """
    
    def __init__(self, data_dir, batch_size, enable_shuffle=False,
                 shuffle_buffer_size=1000, shuffle_seed=0):
        """
        Function for initializing the StreamSampler.
        Inputs:
            data_dir - Directory where the data is stored
            batch_size - Size of the batches
            enable_shuffle - Whether to shuffle the dataset
            shuffle_buffer_size - Size of the buffer for shuffling
            shuffle_seed - Seed of the worker
        """
        
        data_paths = [os.path.join(data_dir, 'behaviors.tsv')]
        self.stream_reader = StreamReader(
            data_paths, 
            batch_size,
            enable_shuffle,
            shuffle_buffer_size
        )

    def __iter__(self):
        """
        Function for iterating over the stream.
        """
        
        self.stream_reader.reset()
        return self

    def __next__(self):
        """
        Function for retrieving the next batch.
        Outputs:
            next_batch - Next batch in the dataset
        """
        
        next_batch = self.stream_reader.get_next()
        if not isinstance(next_batch, np.ndarray) and not isinstance(
                next_batch, tuple):
            raise StopIteration
        return next_batch

    def reach_end(self):
        """
        Function for assessing whether the end of the dataset has been reached.
        Outputs:
            self.stream_reader.endofstream - Boolean indicating the end of the stream 
        """
        
        return self.stream_reader.reach_end()


class StreamReaderTest:
    """
    StreamReader class for streaming the test data.
    """
    
    @tf.autograph.experimental.do_not_convert
    def __init__(self, data_paths, batch_size, shuffle=False, shuffle_buffer_size=1000):
        """
        Function for initializing the StreamReaderTest.
        Inputs:
            data_paths - Paths to the data files
            batch_size - Size of the batches
            shuffle - Whether to shuffle the dataset
            shuffle_buffer_size - Size of the buffer for shuffling
        """
        
        tf.config.experimental.set_visible_devices([], device_type="GPU")
        path_len = len(data_paths)
        dataset = tf.data.Dataset.list_files(data_paths).interleave(
            lambda x: tf.data.TextLineDataset(x),
            cycle_length=path_len,
            block_length=128,
            num_parallel_calls=min(path_len, 64),
        )
        dataset = dataset.interleave(
            lambda x: tf.data.Dataset.from_tensor_slices(
                self._process_record(x)),
            cycle_length=path_len,
            block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        self.dataset = iter(dataset)
        self.session = None
        self.endofstream = False
    
    def _process_record(self, record):
        """
        Function for processing a record in the dataset.
        Inputs:
            record - Line from the dataset
        Outputs:
            sess_poss - Possitive impression
            repeated_record - Record repeated for the number of positive impressions
        """
        
        # iid, uid, time, his, impr
        records = tf.strings.split([record], '\t').values
        sess = tf.strings.split([records[4]], ' ').values  # (num)
        sess_label = tf.strings.split(sess, '-').values

        sess_poss = tf.gather(sess_label, tf.where(tf.equal(sess_label, '1'))-1)
        record = tf.expand_dims(record, axis=0)
        poss_num = tf.size(sess_poss)
        
        test = tf.convert_to_tensor(['test'])
        return test, record

    def reset(self):
        """
        Function for resetting the stream.
        """
        
        self.endofstream = False

    def get_next(self):
        """
        Function for retrieving the next batch.
        Outputs:
            ret - Next batch in the dataset
              None if end of the dataset
        """
        
        try:
            ret = self.dataset.get_next()
        except tf.errors.OutOfRangeError:
            self.endofstream = True
            return None
        return ret

    def reach_end(self):
        """
        Function for assessing whether the end of the dataset has been reached.
        Outputs:
            self.endofstream - Boolean indicating the end of the stream 
        """
        
        return self.endofstream


class StreamSamplerTest:
    """
    StreamSampler class for streaming the test data.
    """
    
    def __init__(self, data_dir, batch_size, enable_shuffle=False,
                 shuffle_buffer_size=1000, shuffle_seed=0):
        """
        Function for initializing the StreamSamplerTest.
        Inputs:
            data_dir - Directory where the data is stored
            batch_size - Size of the batches
            enable_shuffle - Whether to shuffle the dataset
            shuffle_buffer_size - Size of the buffer for shuffling
            shuffle_seed - Seed of the worker
        """
        
        data_paths = [os.path.join(data_dir, 'behaviors.tsv')]
        self.stream_reader = StreamReaderTest(
            data_paths, 
            batch_size, 
            enable_shuffle,
            shuffle_buffer_size)
        
    def __iter__(self):
        """
        Function for iterating over the stream.
        """
        
        if self.stream_reader.endofstream:
          raise StopIteration
        self.stream_reader.reset()
        return self

    def __next__(self):
        """
        Function for retrieving the next batch.
        Outputs:
            next_batch - Next batch in the dataset
        """
        
        next_batch = self.stream_reader.get_next()
        if not isinstance(next_batch, np.ndarray) and not isinstance(
                next_batch, tuple):
            raise StopIteration
        # print(next_batch.shape)
        return next_batch

    def reach_end(self):
        """
        Function for assessing whether the end of the dataset has been reached.
        Outputs:
            self.stream_reader.endofstream - Boolean indicating the end of the stream 
        """
        
        return self.stream_reader.reach_end()