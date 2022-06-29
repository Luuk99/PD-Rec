import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class NRMSAttentionPooling(nn.Module):
    """
    NRMS Attention Pooling class.
    """
    
    def __init__(self, emb_size, hidden_size):
        """
        Function for initializing the NRMSAttentionPooling.
        Inputs:
            emb_size - Embedding size
            hidden_size - Size of the hidden representation
        """
        super(NRMSAttentionPooling, self).__init__()
        
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


class NRMSScaledDotProductAttention(nn.Module):
    """
    NRMS Scaled Dot Product class.
    """
    
    def __init__(self, d_k):
        """
        Function for initializing the NRMSScaledDotProductAttention.
        Inputs:
            d_k - Dimensionality of the key vector
        """
        super(NRMSScaledDotProductAttention, self).__init__()
        
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            Q - Batch of query vectors (batch_size x n_head x candidate_num x d_k)
            K - Batch of key vectors (batch_size x n_head x candidate_num x d_k)
            V - Batch of value vectors (batch_size x n_head x candidate_num x d_v)
            attn_mask - Batch of attention masks (batch_size x n_head x candidate_num)
        Outputs:
            context - Context vectors (batch_size x n_head x candidate_num x d_v)
        """
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)

        if attn_mask is not None:
            scores = scores * attn_mask.unsqueeze(dim=-2)

        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(attn, V)
        return context


class NRMSMultiHeadSelfAttention(nn.Module):
    """
    NRMS Multi Head Self Attention class.
    """
    
    def __init__(self, d_model, n_heads, d_k, d_v):
        """
        Function for initializing the NRMSMultiHeadSelfAttention
        Inputs:
            d_model - Dimensionality of the model
            n_heads - Number of attention heads
            d_k - Dimensionality of the key vector
            d_v - Dimensionality of the value vector
        """
        
        super(NRMSMultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.scaled_dot_product_attn = NRMSScaledDotProductAttention(self.d_k)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Function for initializing the weights
        """
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, mask=None):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            Q - Batch of query vectors (batch_size x candidate_num x d_model)
            K - Batch of key vectors (batch_size x candidate_num x d_model)
            V - Batch of value vectors (batch_size x candidate_num x d_model)
            mask - Batch of attention masks (batch_size x candidate_num)
        Outputs:
            output - Context vectors (batch_size x candidate_num x n_head * d_v)
        """

        batch_size = Q.shape[0]
        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context = self.scaled_dot_product_attn(q_s, k_s, v_s, mask)
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        return output

      
class NRMSNewsEncoder(nn.Module):
    """
    NRMS News Encoder class.
    """
    
    def __init__(self, args, embedding_matrix):
        """
        Function for initializing the NRMSNewsEncoder.
        Inputs:
            args - Parsed arguments
            embedding_matrix - Matrix of pre-trained word embeddings
        """      
        super(NRMSNewsEncoder, self).__init__()
        
        self.embedding_matrix = embedding_matrix
        self.drop_rate = args.drop_rate
        self.dim_per_head = args.news_dim // args.num_attention_heads
        assert args.news_dim == args.num_attention_heads * self.dim_per_head
        self.multi_head_self_attn = NRMSMultiHeadSelfAttention(
            300,
            args.num_attention_heads,
            self.dim_per_head,
            self.dim_per_head
        )
        self.attn = NRMSAttentionPooling(args.news_dim, args.news_query_vector_dim)

    def forward(self, x, mask=None):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            x - Batch of article texts (batch_size x word_num)
            mask - Batch of attention masks (batch_size x word_num)
        Outputs:
            news_vec - News vector representations (batch_size x news_dim)
        """
        
        word_vecs = F.dropout(self.embedding_matrix(x.long()),
                              p=self.drop_rate,
                              training=self.training)
        multihead_text_vecs = self.multi_head_self_attn(word_vecs, word_vecs, word_vecs, mask)
        multihead_text_vecs = F.dropout(multihead_text_vecs,
                                        p=self.drop_rate,
                                        training=self.training)
        news_vec = self.attn(multihead_text_vecs, mask)
        return news_vec


class NRMSUserEncoder(nn.Module):
    """
    NRMS User Encoder class.
    """
    
    def __init__(self, args):
        """
        Function for initializing the NRMSUserEncoder.
        Inputs:
            args - Parsed arguments
        """
        super(NRMSUserEncoder, self).__init__()
        
        self.args = args
        self.dim_per_head = args.news_dim // args.num_attention_heads
        assert args.news_dim == args.num_attention_heads * self.dim_per_head
        self.multi_head_self_attn = NRMSMultiHeadSelfAttention(args.news_dim, args.num_attention_heads, self.dim_per_head, self.dim_per_head)
        self.attn = NRMSAttentionPooling(args.news_dim, args.user_query_vector_dim)
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
        news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs, log_mask)
        user_vec = self.attn(news_vecs, log_mask)
        
        return user_vec


class NRMSModel(torch.nn.Module):
    """
    NRMS Model class.
    """
    
    def __init__(self, args, embedding_matrix, **kwargs):
        """
        Function for initializing the NRMSUserEncoder.
        Inputs:
            args - Parsed arguments
            embedding_matrix - Matrix of pre-trained word embeddings
            kwargs - Optional additional arguments
        """
        super(NRMSModel, self).__init__()
        
        self.args = args
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=False,
                                                      padding_idx=0)

        self.news_encoder = NRMSNewsEncoder(args, word_embedding)
        self.user_encoder = NRMSUserEncoder(args)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, candidate, history, history_mask, targets=None):
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
        candidate = candidate.view(-1, input_ids_length)
        candidate_news_vecs = self.news_encoder(candidate)
        candidate_news_vecs = candidate_news_vecs.view(-1, 1 + self.args.npratio, self.args.news_dim)
        
        # Calculate the history news embeddings
        history_news = history.view(-1, input_ids_length)
        history_news_vecs = self.news_encoder(history_news)
        history_news_vecs = history_news_vecs.view(-1, self.args.user_log_length, self.args.news_dim)
        
        # Calculate the user embeddings
        user_vec = self.user_encoder(history_news_vecs, history_mask)
        
        # Score the candidate documents
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        # Calculate the loss if targets are given
        if targets is not None:
          # Calculate the loss
          loss = self.loss_fn(score, targets)
          return loss, score
        else:
          return score