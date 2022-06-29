import torch
import torch.nn as nn


class FastformerAttentionPooling(nn.Module):
    """
    Fastformer Attention Pooling class.
    """
    
    def __init__(self, config):
        """
        Function for initializing the FastformerAttentionPooling.
        Inputs:
            config - Config for the Fastformer model
        """
        super(FastformerAttentionPooling, self).__init__()
        
        self.config = config
        self.att_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.att_fc2 = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        """
        Function for initializing the weights
        Inputs:
            module - Module to initialize the weights of
        """
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()           
                
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
        x = torch.reshape(x, (bz, -1))  
        return x


class FastSelfAttention(nn.Module):
    """
    Fast SelfAttention class.
    """
    
    def __init__(self, config):
        """
        Function for initializing the FastSelfAttention
        Inputs:
            config - Config for the Fastformer model
        """      
        super(FastSelfAttention, self).__init__()
        
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.attention_head_size = int(config.hidden_size /config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim= config.hidden_size
        
        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        
        self.apply(self.init_weights)

    def init_weights(self, module):
        """
        Function for initializing the weights
        Inputs:
            module - Module to initialize the weights of
        """
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
                
    def transpose_for_scores(self, x):
        """
        Function for transposing x.
        Inputs:
            x - Vector tensor
        """
        
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            hidden_states - Hidden states of the model
            attention_mask - Batch of attention masks
        Outputs:
            weighted_value - Applied fast attention
        """
        
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        # add attention mask
        query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer=mixed_key_layer* pooled_query_repeat
        
        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        
        # add attention mask
        query_key_score +=attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
      
        return weighted_value

      
class FastAttention(nn.Module):
    """
    Fast Attention class.
    """
    
    def __init__(self, config):
        """
        Function for initializing the FastAttention
        Inputs:
            config - Config for the Fastformer model
        """ 
        super(FastAttention, self).__init__()
        
        self.self = FastSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            input_tensor - Vector tensor
            attention_mask - Batch of attention masks
        Outputs:
            attention_output - Attention output of the vector
        """
        
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

      
class FastformerLayer(nn.Module):
    """
    Fastformer Layer class.
    """
    
    def __init__(self, config):
        """
        Function for initializing the FastformerLayer
        Inputs:
            config - Config for the Fastformer model
        """ 
        super(FastformerLayer, self).__init__()
        
        self.attention = FastAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            hidden_states - Vector tensor
            attention_mask - Batch of attention masks
        Outputs:
            layer_output - Output of the fastformer layer
        """
        
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
  
  
class FastformerEncoder(nn.Module):
    """
    Fastformer Encoder class.
    """
    
    def __init__(self, config, pooler_count=1):
        """
        Function for initializing the FastformerLayer
        Inputs:
            config - Config for the Fastformer model
            pooler_count - Count of the poolers
        """ 
        super(FastformerEncoder, self).__init__()
        
        self.config = config
        self.encoders = nn.ModuleList([FastformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.ModuleList()
        if config.pooler_type == 'weightpooler':
            for _ in range(pooler_count):
                self.poolers.append(FastformerAttentionPooling(config))

        self.apply(self.init_weights)

    def init_weights(self, module):
        """
        Function for initializing the weights
        Inputs:
            module - Module to initialize the weights of
        """
        
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_embs, attention_mask, pooler_index=0):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            input_embs - Batch of input embeddings
            attention_mask - Batch of attention masks
            pooler_index - Index of the pooler
        Outputs:
            all_hidden_states[-1] - Last hidden states of the model
        """
        
        #input_embs: batch_size, seq_len, emb_dim
        #attention_mask: batch_size, seq_len, emb_dim

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        batch_size, seq_length, emb_dim = input_embs.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embs.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embs + position_embeddings
        embeddings = self.dropout(embeddings)
        all_hidden_states = [embeddings]

        for i, layer_module in enumerate(self.encoders):
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)
        return all_hidden_states[-1]

      
class FastformerModel(torch.nn.Module):
    """
    Fastformer Model class.
    """

    def __init__(self, config, word_dict, embedding_matrix):
        """
        Function for initializing the FastformerModel
        Inputs:
            config - Config for the Fastformer model
            word_dict - Word dictionary
            embedding_matrix - Matrix of pre-trained embeddings
        """ 
        super(FastformerModel, self).__init__()
        
        self.config = config
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        self.word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding, freeze=False, padding_idx=0)
        self.word_relu = nn.ReLU()
        self.word_dim_reduction = nn.Linear(300, config.hidden_size)
        self.fastformer_model = FastformerEncoder(config)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        """
        Function for initializing the weights
        Inputs:
            module - Module to initialize the weights of
        """
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, input_ids):
        """
        Function for performing a forward pass on a batch.
        Inputs:
            input_ids - Batch of input ids
        Outputs:
            text_vec - Batch of text vectors
        """
        
        mask=input_ids.bool().float()
        embds = self.word_embedding(input_ids)
        embds = self.word_relu(embds)
        embds = self.word_dim_reduction(embds)
        text_vec = self.fastformer_model(embds, mask)
        return text_vec