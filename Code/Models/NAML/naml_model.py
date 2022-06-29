import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class NAMLAttentionPooling(nn.Module):
    """
    NAML Attention Pooling class.
    """
    
    def __init__(self, emb_size, hidden_size):
        """
        Function for initializing the NAMLAttentionPooling.
        Inputs:
            emb_size - Embedding size
            hidden_size - Size of the hidden representation
        """
        super(NAMLAttentionPooling, self).__init__()
        
        self.att_fc1 = nn.Linear(emb_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            x - Batch of vectors (batch_size x candidate_size x emb_dim)
            attn_mask - Batch of attention masks (batch_size x candidate_size)
        Outputs:
            x - Attention pooled batch (batch_size, emb_dim)
        """
        
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)

        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)

        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)
        return x


class NAMLNewsEncoder(nn.Module):
    """
    NAML News Encoder class.
    """
    
    def __init__(self, args, embedding_matrix, num_category, num_subcategory):
        """
        Function for initializing the NAMLNewsEncoder.
        Inputs:
            args - Parsed arguments
            embedding_matrix - Matrix of pre-trained word embeddings
            num_category - Number of categories
            num_subcategory - Number of subcategories
        """
        super(NAMLNewsEncoder, self).__init__()
        
        self.embedding_matrix = embedding_matrix
        self.drop_rate = args.drop_rate
        self.num_words_title = args.num_words_title
        self.num_words_abstract = args.num_words_abstract
        self.dataset = args.dataset
        
        # Initialize category embedding
        self.category_emb = nn.Embedding(num_category + 1, 100, padding_idx=0)
        self.category_dense = nn.Linear(100, args.news_dim)
        
        # Initialize subcategory embedding
        if args.dataset == 'MIND':
          self.subcategory_emb = nn.Embedding(num_subcategory + 1, 100, padding_idx=0)
          self.subcategory_dense = nn.Linear(100, args.news_dim)
        
        # Initialize the final pooling layers
        self.final_attn = NAMLAttentionPooling(args.news_dim, args.news_query_vector_dim)
        self.cnn = nn.Conv1d(
            in_channels=300,
            out_channels=args.news_dim,
            kernel_size=3,
            padding=1
        )
        self.attn = NAMLAttentionPooling(args.news_dim, args.news_query_vector_dim)

    def forward(self, x, mask=None):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            x - Batch of article texts (batch_size x word_num)
            mask - Batch of attention masks (batch_size x word_num)
        Outputs:
            news_vecs - News vector representations (batch_size x news_dim)
        """
        
        title = torch.narrow(x, -1, 0, self.num_words_title).long()
        word_vecs = F.dropout(self.embedding_matrix(title),
                              p=self.drop_rate,
                              training=self.training)
        context_word_vecs = self.cnn(word_vecs.transpose(1, 2)).transpose(1, 2)
        title_vecs = self.attn(context_word_vecs, mask)
        all_vecs = [title_vecs]

        start = self.num_words_title + self.num_words_abstract
        category = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
        category_vecs = self.category_dense(self.category_emb(category))
        all_vecs.append(category_vecs)
        
        if self.dataset == 'MIND':
          start += 1
          subcategory = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
          subcategory_vecs = self.subcategory_dense(self.subcategory_emb(subcategory))
          all_vecs.append(subcategory_vecs)

        if len(all_vecs) == 1:
            news_vecs = all_vecs[0]
        else:
            all_vecs = torch.stack(all_vecs, dim=1)
            news_vecs = self.final_attn(all_vecs)
        return news_vecs


class NAMLUserEncoder(nn.Module):
    """
    NAML User Encoder class.
    """
    
    def __init__(self, args):
        """
        Function for initializing the NAMLUserEncoder.
        Inputs:
            args - Parsed arguments
        """
        super(NAMLUserEncoder, self).__init__()
        
        self.attn = NAMLAttentionPooling(args.news_dim, args.user_query_vector_dim)
        self.pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)

    def forward(self, news_vecs, log_mask=None):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            news_vecs - Batch of historical news vectors (batch_size x history_num x news_dim)
            log_mask - Batch of attention masks (batch_size x history_num)
        Outputs:
            user_vec - User vector representations (batch_size x news_dim)
        """
        
        bz = news_vecs.shape[0]
        user_vec = self.attn(news_vecs, log_mask)
        return user_vec


class NAMLModel(torch.nn.Module):
    """
    NAML Model class.
    """
    
    def __init__(self, args, embedding_matrix, num_category, num_subcategory):
        """
        Function for initializing the NAMLUserEncoder.
        Inputs:
            args - Parsed arguments
            embedding_matrix - Matrix of pre-trained word embeddings
            num_category - Number of categories
            num_subcategory - Number of subcategories
        """
        super(NAMLModel, self).__init__()
        
        self.args = args
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=False,
                                                      padding_idx=0)

        self.news_encoder = NAMLNewsEncoder(args, word_embedding, num_category, num_subcategory)
        self.user_encoder = NAMLUserEncoder(args)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, candidate, history, history_mask, targets):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            candidate - Batch of candidate news articles (batch_size x 1+K x num_word_title)
            history - Batch of historical news articles (batch_size x history_length x num_word_title)
            history_mask - Batch of attention masks (batch_size, history_length)
            targets - Batch of target labels (batch_size, 1+K)
        Outputs:
            loss - Loss of the model
            score - Scores of the candidate articles
        """
        
        # Calculate the candidate news embeddings
        input_ids_length = candidate.size(2)
        candidate_news = candidate.reshape(-1, input_ids_length)
        candidate_news_vecs = self.news_encoder(candidate_news)
        candidate_news_vecs = candidate_news_vecs.reshape(-1, 1 + self.args.npratio, self.args.news_dim)
        
        # Calculate the historynews embeddings
        history_news = history.reshape(-1, input_ids_length)
        history_news_vecs = self.news_encoder(history_news)
        history_news_vecs = history_news_vecs.reshape(-1, self.args.user_log_length, self.args.news_dim)
        
        # Calculate the user embeddings
        user_vec = self.user_encoder(history_news_vecs, history_mask)
        
        # Score the candidate documents
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        # Calculate the loss if targets are given
        if targets is not None:
          loss = self.loss_fn(score, targets)
          return loss, score
        else:
          return score