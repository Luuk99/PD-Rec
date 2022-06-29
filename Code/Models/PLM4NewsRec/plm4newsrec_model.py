import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PLM4NewsRecAdditiveAttention(nn.Module):
    """
    PLM4NewsRec Additive Attention class.
    """
    
    def __init__(self, d_h, hidden_size=200):
        """
        Function for initializing the PLM4NewsRecAdditiveAttention.
        Inputs:
            d_h - Last dimension of the input
            hidden_size - Size of the hidden representation
        """
        super(PLM4NewsRecAdditiveAttention, self).__init__()
        
        self.att_fc1 = nn.Linear(d_h, hidden_size)
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
        
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x


class PLM4NewsRecScaledDotProductAttention(nn.Module):
    """
    PLM4NewsRec Scaled Dot Product class.
    """
    
    def __init__(self, d_k):
        """
        Function for initializing the PLM4NewsRecScaledDotProductAttention.
        Inputs:
            d_k - Dimensionality of the key vector
        """
        super(PLM4NewsRecScaledDotProductAttention, self).__init__()
        
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
            attn - Attention over the input
        """
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        
        if attn_mask is not None:
            scores = scores * attn_mask
            
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(attn, V)
        return context, attn


class PLM4NewsRecMultiHeadAttention(nn.Module):
    """
    PLM4NewsRec Multi Head Attention class.
    """
    
    def __init__(self, d_model, n_heads, d_k, d_v):
        """
        Function for initializing the PLM4NewsRecMultiHeadAttention
        Inputs:
            d_model - Dimensionality of the model
            n_heads - Number of attention heads
            d_k - Dimensionality of the key vector
            d_v - Dimensionality of the value vector
        """      
        super(PLM4NewsRecMultiHeadAttention, self).__init__()
        
        self.d_model = d_model  # 300
        self.n_heads = n_heads  # 20
        self.d_k = d_k  # 20
        self.d_v = d_v  # 20

        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 300, 400

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
        
        batch_size, seq_len, _ = Q.shape

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len) #  [bz, seq_len, seq_len]
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [bz, 20, seq_len, seq_len]

        context, attn = PLM4NewsRecScaledDotProductAttention(self.d_k)(
            q_s, k_s, v_s, mask)  # [bz, 20, seq_len, 20]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)  # [bz, seq_len, 400]
        #         output = self.fc(context)
        return context  #self.layer_norm(output + residual)


class PLM4NewsRecTextEncoder(torch.nn.Module):
    """
    PLM4NewsRec Text Encoder class.
    """
    
    def __init__(self, news_encoder_model, token_embedding_dim, num_attention_heads, query_vector_dim, dropout_rate, news_encoder_class):
        """
        Function for initializing the PLM4NewsRecTextEncoder.
        Inputs:
            news_encoder_model - Model used for encoding the news articles
            token_embedding_dim - Dimensionality of the tokens
            num_attention_heads - Number of attention heads
            query_vector_dim - Dimensionality of the news query vector
            dropout_rate - Dropout rate during training
            news_encoder_class - Name of the news encoder model
        """
        super(PLM4NewsRecTextEncoder, self).__init__()
        
        self.news_encoder_model = news_encoder_model
        self.news_encoder_class = news_encoder_class
        self.dropout_rate = dropout_rate
        self.text_vector_dropout = nn.Dropout(p=self.dropout_rate)
        self.multihead_attention = PLM4NewsRecMultiHeadAttention(token_embedding_dim, num_attention_heads, 20, 20)
        self.multihead_text_vector_dropout = nn.Dropout(p=self.dropout_rate)
        self.additive_attention = PLM4NewsRecAdditiveAttention(num_attention_heads*20, query_vector_dim)

    def forward(self, text, mask=None):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            text - Batch of tokenized texts
            mask - Mask to use if given
              None if no mask is provided
        Outputs:
            text_embeddings - Batch of text embeddings
        """
        
        # batch_size, num_words_text
        batch_size, num_words = text.shape
        if self.news_encoder_class == 'bert':
          num_words = num_words // 3
          text_ids = torch.narrow(text, 1, 0, num_words)
          text_type = torch.narrow(text, 1, num_words, num_words)
          text_attmask = torch.narrow(text, 1, num_words*2, num_words)
          word_emb = self.news_encoder_model(text_ids, text_type, text_attmask)[0]
        else:
          word_emb = self.news_encoder_model(text)
        text_vector = self.text_vector_dropout(word_emb)
        # batch_size, num_words_text, token_embedding_dim
        multihead_text_vector = self.multihead_attention(
            text_vector, text_vector, text_vector, mask)
        multihead_text_vector = self.multihead_text_vector_dropout(multihead_text_vector)
        # batch_size, token_embedding_dim
        text_embeddings = self.additive_attention(multihead_text_vector, mask)
        return text_embeddings


class PLM4NewsRecElementEncoder(torch.nn.Module):
    """
    PLM4NewsRec Element Encoder class.
    """
    
    def __init__(self, num_elements, embedding_dim, enable_gpu=True):
        """
        Function for initializing the PLM4NewsRecElementEncoder.
        Inputs:
            num_elements - Total number of elements
            embedding_dim - Dimensionality of the embedding
            enable_gpu - Whether to use the GPU
        """
        
        super(PLM4NewsRecElementEncoder, self).__init__()
        self.enable_gpu = enable_gpu
        self.embedding = nn.Embedding(num_elements,
                                      embedding_dim,
                                      padding_idx=0)

    def forward(self, element):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            element - Batch of elements
        Outputs:
            element_embeddings - Batch of element embeddings
        """
        
        # Calculate the element embeddings
        element_embeddings = self.embedding(
            (element.cuda() if self.enable_gpu else element).long())
        
        # Return the element embeddings
        return element_embeddings


class PLM4NewsRecNewsEncoder(torch.nn.Module):
    """
    PLM4NewsRec News Encoder class.
    """
    
    def __init__(self, args, news_encoder_model, token_embedding_dim, category_dict_size, subcategory_dict_size):
        """
        Function for initializing the PLM4NewsRecNewsEncoder.
        Inputs:
            args - Parsed arguments
            news_encoder_model - Model used for encoding the news articles
            token_embedding_dim - Dimensionality of the tokens
            category_dict_size - Number of distinct categories
            subcategory_dict_size - Number of distinct subcategories
        """
        
        super(PLM4NewsRecNewsEncoder, self).__init__()
        self.args = args
        
        # Check the lengths of the texts
        if args.news_encoder_model == 'bert':
          text_multiple = 3
        else:
          text_multiple = 1
        
        # Convert attributes to their lengths
        self.attributes2length = {
            'title': args.num_words_title * text_multiple,
            'abstract': args.num_words_abstract * text_multiple,
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
        
        # Initialize the encoders for the text
        text_encoders_candidates = ['title', 'abstract']
        self.text_encoders = nn.ModuleDict({
            'title':
            PLM4NewsRecTextEncoder(news_encoder_model,
                        token_embedding_dim,
                        args.num_attention_heads, args.news_query_vector_dim,
                        args.drop_rate, args.news_encoder_model)
        })
        self.newsname=[name for name in set(text_encoders_candidates)]
        
        # Initialize the encoders for the category and subcategory
        name2num = {
            "category": category_dict_size + 1,
            "subcategory": subcategory_dict_size + 1
        }
        if args.dataset == 'MIND':
          element_encoders_candidates = ['category', 'subcategory']
        else:
          element_encoders_candidates = ['category']
        self.element_encoders = nn.ModuleDict({
            name: PLM4NewsRecElementEncoder(name2num[name], 
                                args.num_attention_heads * 20,
                                 args.enable_gpu)
            for name in (set(element_encoders_candidates))
        })
        
        # Initialize the layer for dimensionality reduction
        self.reduce_dim_linear = nn.Linear(args.num_attention_heads * 20,
                                           args.news_dim)

    def forward(self, news):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            news - Batch of news articles
        Outputs:
            news_embeddings - Batch of news embeddings
        """
        
        # Compute the embeddings of the news article contents
        text_embeddings = [
            self.text_encoders['title'](
                torch.narrow(news, 1, self.attributes2start[name],
                             self.attributes2length[name]))
            for name in self.newsname
        ]
        
        # Compute the embeddings of the category and subcategory
        element_embeddings = [
            encoder(
                torch.narrow(news, 1, self.attributes2start[name],
                             self.attributes2length[name]).squeeze(dim=1))
            for name, encoder in self.element_encoders.items()
        ]
        
        # Combine the text and element embeddings
        news_embeddings = text_embeddings + element_embeddings
        if len(news_embeddings) == 1:
            news_embeddings = news_embeddings[0]
        else:
            news_embeddings = torch.mean(torch.stack(news_embeddings, dim=1), dim=1)
        
        # Reduce the dimensionality
        news_embeddings = self.reduce_dim_linear(news_embeddings)
        
        # Return the news embeddings
        return news_embeddings


class PLM4NewsRecUserEncoder(torch.nn.Module):
    """
    PLM4News User Encoder class.
    """
    
    def __init__(self, args):
        """
        Function for initializing the PLM4NewsUserEncoder.
        Inputs:
            args - Parsed arguments
        """
        
        super(PLM4NewsRecUserEncoder, self).__init__()
        self.args = args
        
        # Initialize the additive attention
        self.additive_attention = PLM4NewsRecAdditiveAttention(args.news_dim, args.user_query_vector_dim)      
    
    def forward(self, history_embeddings, history_attention_mask):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            history_embddings - Embeddings of the users historical news
            history_attention_mask - Attention mask for the historical news articles
        Outputs:
            user_embeddings - Embeddings of the users
        """
        
        # Apply additive attention
        user_embeddings = self.additive_attention(history_embeddings, history_attention_mask)
        
        # Return the user embeddings
        return user_embeddings

    
class PLM4NewsRecModel(torch.nn.Module):
    """
    PLM4News Model class.
    """
    
    def __init__(self, args, news_encoder_model, token_embedding_dim, category_dict_size=0, subcategory_dict_size=0):
        """
        Function for initializing the PLM4NewsModel.
        Inputs:
            args - Parsed arguments
            news_encoder_model - Model used for encoding the news articles
            token_embedding_dim - Dimensionality of the tokens
            category_dict_size - Number of distinct categories
            subcategory_dict_size - Number of distinct subcategories
        """
        
        super(PLM4NewsRecModel, self).__init__()
        self.args = args
        
        # Initialize the news encoder
        self.news_encoder = PLM4NewsRecNewsEncoder(args, news_encoder_model, token_embedding_dim, category_dict_size,
                                        subcategory_dict_size)
        
        # Initialize the user encoder
        self.user_encoder = PLM4NewsRecUserEncoder(args)
        
        # Initialize the loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, candidate_input_ids, history_input_ids, history_attention_mask, targets=None):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            candidate_input_ids - Token ids for the candidate news articles
            history_input_ids - Token ids for the historical news articles
            history_attention_mask - Attention mask for the historical news articles
            targets - Labels for the inputs
        Outputs:
            loss - Loss of the batch
            candidate_scores - Scores of the candidate news articles
        """
        
        # Calculate the candidate news embeddings
        input_ids_length = candidate_input_ids.size(2)
        candidate_input_ids = candidate_input_ids.view(-1, input_ids_length)
        candidate_embeddings = self.news_encoder(candidate_input_ids)
        candidate_embeddings = candidate_embeddings.view(-1, 1 + self.args.npratio, self.args.news_dim)

        # Calculate the history news embeddings
        history_input_ids = history_input_ids.view(-1, input_ids_length)
        history_embeddings = self.news_encoder(history_input_ids)
        history_embeddings = history_embeddings.view(-1, self.args.user_log_length, self.args.news_dim)
        
        # Calculate the user embeddings
        user_embeddings = self.user_encoder(history_embeddings, history_attention_mask)

        # Score the candidate documents
        candidate_scores = torch.bmm(candidate_embeddings, user_embeddings.unsqueeze(-1)).squeeze(dim=-1)
        
        # Calculate the loss if targets are given
        if targets is not None:
            # Calculate the loss
            loss = self.criterion(candidate_scores, targets)
            return loss, candidate_scores
        else:
            return candidate_scores