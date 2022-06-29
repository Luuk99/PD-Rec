import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

      
class LSTURAdditiveAttention(nn.Module):
    """
    LSTUR Additive Attention class.
    """
    
    def __init__(self, emb_size, hidden_size):
        """
        Function for initializing the LSTURAdditiveAttention.
        Inputs:
            emb_size - Embedding size
            hidden_size - Size of the hidden representation
        """
        super(LSTURAdditiveAttention, self).__init__()
        
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

      
class LSTURNewsEncoder(nn.Module):
    """
    LSTUR News Encoder class.
    """
    
    def __init__(self, args, embedding_matrix, category_dict_size=0, subcategory_dict_size=0, token_embedding_dim=300):
        """
        Function for initializing the LSTURNewsEncoder.
        Inputs:
            args - Parsed arguments
            embedding_matrix - Matrix of pre-trained word embeddings
            category_dict_size - Number of distinct categories
            subcategory_dict_size - Number of distinct subcategories
            token_embedding_dim - Dimensionality of the tokens
        """      
        super(LSTURNewsEncoder, self).__init__()
        
        self.embedding_matrix = embedding_matrix
        self.args = args
        
        # Convert attributes to their lengths
        self.attributes2length = {
            'title': args.num_words_title,
            'abstract': args.num_words_abstract,
            'category': 1,
            'subcategory': 1
        }
        
        # Convert lengths to their starting index
        self.attributes2start = {
            key: sum(
                list(self.attributes2length.values())
                [:list(self.attributes2length.keys()).index(key)])
            for key in self.attributes2length.keys()
        }
        
        # Initialize dropout
        self.dropout = nn.Dropout(p=self.args.drop_rate)
        
        # Initialize the category embeddings
        self.category_embedding = nn.Embedding(category_dict_size + 1, token_embedding_dim, padding_idx=0)
        if args.dataset == 'MIND':
          self.subcategory_embedding = nn.Embedding(subcategory_dict_size + 1, token_embedding_dim, padding_idx=0)
        
        # Initialize the CNN
        self.cnn = nn.Conv1d(
            in_channels=300,
            out_channels=300,
            kernel_size=3,
            padding=1
        )
        
        # Initialize the additive attention
        self.attention = LSTURAdditiveAttention(300, args.news_query_vector_dim)

    def forward(self, x, mask=None):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            x - Batch of article texts (batch_size x word_num)
            mask - Batch of attention masks (batch_size x word_num)
        Outputs:
            news_vec - News vector representations (batch_size x news_dim)
        """
        
        # Handle the title
        title_vec = torch.narrow(x, 1, self.attributes2start['title'], self.attributes2length['title'])
        title_vec = self.embedding_matrix(title_vec)
        title_vec = self.dropout(title_vec)
        title_vec = self.cnn(title_vec.transpose(1, 2)).transpose(1, 2)
        title_vec = F.relu(title_vec)
        title_vec = self.dropout(title_vec)
        title_vec = self.attention(title_vec, mask)
        
        # Handle the category and subcategory
        category_vec = torch.narrow(x, 1, self.attributes2start['category'], self.attributes2length['category'])
        category_vec = self.category_embedding(category_vec.squeeze(dim=1).long())
        if self.args.dataset == 'MIND':
          subcategory_vec = torch.narrow(x, 1, self.attributes2start['subcategory'], self.attributes2length['subcategory'])
          subcategory_vec = self.subcategory_embedding(subcategory_vec.squeeze(dim=1).long())
        
        # Concatenate into a single news vector
        if self.args.dataset == 'MIND':
          news_vec = torch.cat([category_vec, subcategory_vec, title_vec], dim=-1)
        else:
          news_vec = torch.cat([category_vec, title_vec], dim=-1)
        
        return news_vec


class LSTURUserEncoder(nn.Module):
    """
    LSTUR User Encoder class.
    """
    
    def __init__(self, args):
        """
        Function for initializing the NRMSUserEncoder.
        Inputs:
            args - Parsed arguments
        """
        super(LSTURUserEncoder, self).__init__() 
        self.args = args
        
        # Initialize the GRU
        if self.args.dataset == 'MIND':
          self.gru = nn.GRU(3 * 300, 3 * 300)
        else:
          self.gru = nn.GRU(2 * 300, 2 * 300)

    def forward(self, user, news_vecs, log_mask=None):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            user - User batch
            news_vecs - Batch of historical news vectors (batch_size x history_num x news_dim)
            log_mask - Batch of attention masks (batch_size x history_num)
        Outputs:
            user_vec - User vector representations (batch_size x news_dim)
        """
        
        # Calculate the clicked news length based on the log mask
        clicked_news_length = torch.sum(log_mask, dim=-1)
        clicked_news_length[clicked_news_length == 0] = 1
        
        # Pack the padded sequence
        packed_sequence = pack_padded_sequence(news_vecs, clicked_news_length.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass through the gru
        _, last_hidden = self.gru(packed_sequence, user.unsqueeze(dim=0))
        user_vec = last_hidden.squeeze(dim=0)
        
        return user_vec


class LSTURModel(torch.nn.Module):
    """
    LSTUR Model class.
    """
    
    def __init__(self, args, embedding_matrix, category_dict_size=0, subcategory_dict_size=0, num_users=0, **kwargs):
        """
        Function for initializing the LSTURModel.
        Inputs:
            args - Parsed arguments
            embedding_matrix - Matrix of pre-trained word embeddings
            category_dict_size - Number of distinct categories
            subcategory_dict_size - Number of distinct subcategories
            num_users - Number of users
            kwargs - Optional additional arguments
        """
        super(LSTURModel, self).__init__()
        self.args = args
        
        # Check which dataset is used (if subcategory is included)
        if self.args.dataset == 'MIND':
          self.embedding_multiplier = 3
        else:
          self.embedding_multiplier = 2
        
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=False,
                                                      padding_idx=0)

        self.news_encoder = LSTURNewsEncoder(args, word_embedding, category_dict_size, subcategory_dict_size)
        self.user_embedding = nn.Embedding(num_users + 1, self.embedding_multiplier * 300, padding_idx=0)
        self.user_dropout = nn.Dropout2d(self.args.drop_rate)
        self.user_encoder = LSTURUserEncoder(args)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, candidate, history, history_mask, user, targets=None):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            candidate - Batch of candidate news articles (batch_size x 1+K x num_word_title)
            history - Batch of historical news articles (batch_size x history_length x num_word_title)
            history_mask - Batch of attention masks (batch_size, history_length)
            user - Batch of user ids
            targets - Batch of target labels (batch_size, 1+K)
        Outputs:
            loss - Loss of the model
            score - Scores of the candidate articles
        """
        
        # Calculate the candidate news embeddings
        input_length = candidate.size(2)
        candidate = candidate.view(-1, input_length)
        candidate_news_vecs = self.news_encoder(candidate)
        candidate_news_vecs = candidate_news_vecs.view(-1, 1 + self.args.npratio, self.embedding_multiplier * 300)
        
        # Calculate the history news embeddings
        user = self.user_dropout(self.user_embedding(user.long()).unsqueeze(dim=0)).squeeze(dim=0)
        history = history.view(-1, input_length)
        history_news_vecs = self.news_encoder(history)
        history_news_vecs = history_news_vecs.view(-1, self.args.user_log_length, self.embedding_multiplier * 300)
        
        # Calculate the user embeddings
        user_vec = self.user_encoder(user, history_news_vecs, history_mask)
        
        # Score the candidate documents
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        # Calculate the loss if targets are given
        if targets is not None:
          # Calculate the loss
          loss = self.loss_fn(score, targets)
          return loss, score
        else:
          return score