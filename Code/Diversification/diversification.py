import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
from scipy.special import softmax


def rank_recommendations(scores, labels, embeddings, news_ids):
    """
    Function for ranking the recommendations.
    Inputs:
        scores - Scores of the candidate news articles
        labels - Truth labels for the candidate news articles
        embeddings - Candidate news embeddings
        news_ids - Candidate news ids
    Outputs:
        scores - Ordered scores of the candidate news articles
        labels - Ordered truth labels for the candidate news articles
        embeddings - Ordered candidate news embeddings
        news_ids - Ordered candidate news ids
    """
    
    # Order the scores, labels and embeddings
    order = np.argsort(scores)[::-1]
    scores = np.take(scores, order)
    labels = np.take(labels, order)
    embeddings = np.take(embeddings, order, axis=0)
    news_ids = np.take(news_ids, order)
    
    # Return the ordered scores, labels, embeddings and news ids
    return scores, labels, embeddings, news_ids


def diversity_reranking(history_news_vecs, history_log_mask, candidate_news_vecs, scores):
    """
    Function for calculating a diversified score.
    Inputs:
        history_news_vecs - History news embeddings
        history_log_mask - History log mask
        candidate_news_vecs - Candidate news embeddings
        scores - Relevancy scores
    Outputs:
        diverse_scores - Diversified scores
    """
    
    # Remove the masked vectors
    history_length = int(np.sum(history_log_mask))
    history_news_vecs = history_news_vecs[-history_length:]
    
    # If there is only one item in the history, we use a default value of 0.5
    if history_length < 2:
      history_similarity = 0.5
    else:  
      # Calculate the history similarity
      history_similarity_matrix = cosine_similarity(X=history_news_vecs)
      history_similarity = (history_similarity_matrix + 1) / 2
      upper_right = np.triu_indices(history_similarity_matrix.shape[0], k=1)
      history_similarity = np.mean(history_similarity_matrix[upper_right])
      
      #history_similarity = euclidean_distances(history_news_vecs, history_news_vecs)
      #upper_right = np.triu_indices(history_similarity.shape[0], k=1)
      #history_similarity = 1 / (1 + history_similarity)
      #history_similarity = np.mean(history_similarity[upper_right])
    
    # Calculate the candidate diversities
    candidate_diversity = (1 - cosine_similarity(X=candidate_news_vecs)) / 2
    candidate_diversity = np.sum(candidate_diversity, axis=1) / (np.shape(candidate_diversity)[1] - 1)
    #candidate_diversity = euclidean_distances(candidate_news_vecs, candidate_news_vecs)
    #candidate_diversity = 1 - (1 / (1 + candidate_diversity))
    #candidate_diversity = np.array(candidate_diversity).mean(axis=0)
    
    # Softmax the scores
    scores = softmax(scores)
    
    # Calculate the diverse scores
    diversified_scores = (history_similarity * candidate_diversity) + ((1 - history_similarity) * scores)
    
    # Return the diversified scores, history similarity and candidate diversity
    return diversified_scores, history_similarity, candidate_diversity
    

def user_similarity(history_news_vecs, history_log_mask):
    """
    Function for calculating the user similarity.
    Inputs:
        history_news_vecs - History news embeddings
        history_log_mask - History log mask
    Outputs:
        history_similarity - Similarity of the users history
    """
    
    # Remove the masked vectors
    history_length = int(np.sum(history_log_mask))
    history_news_vecs = history_news_vecs[-history_length:]
    
    # If there is only one item in the history, we use a default value of 0.5
    if history_length < 2:
      history_similarity = 0.5
    else:  
      # Calculate the history similarity
      history_similarity = cosine_similarity(X=history_news_vecs)
      upper_right = np.triu_indices(history_similarity.shape[0], k=1)
      history_similarity = (history_similarity + 1) / 2
      history_similarity = np.mean(history_similarity[upper_right])
    
    # Return the history similarity
    return history_similarity
  
  
def candidate_diversity(candidate_news_vecs):
    """
    Function for calculating diversity of the candidates.
    Inputs:
        candidate_news_vecs - Candidate news embeddings
    Outputs:
        candidate_div - Average diversity of the candidates (intra-list distance)
    """
    
    # Calculate the candidate diversities
    candidate_diversity = (1 - cosine_similarity(X=candidate_news_vecs)) / 2
    upper_right = np.triu_indices(candidate_diversity.shape[0], k=1)
    candidate_diversity = np.mean(candidate_diversity[upper_right])
    
    # Return the candidate diversity
    return candidate_diversity