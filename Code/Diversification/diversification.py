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


def diversity_reranking(args, history_news_vecs, history_log_mask, candidate_news_vecs, scores):
    """
    Function for calculating a diversified score.
    Inputs:
        args - Parsed arguments
        history_news_vecs - History news embeddings
        history_log_mask - History log mask
        candidate_news_vecs - Candidate news embeddings
        scores - Relevancy scores
    Outputs:
        diverse_scores - Diversified scores

        diversified_scores -  Diversified scrores
        history_similarity - User's history similarity
        candidate_diversity - Candidate articles' diversity
    """
    
    # Remove the masked vectors
    history_length = int(np.sum(history_log_mask))
    history_news_vecs = history_news_vecs[-history_length:]
    
    # If there is only one item in the history, we use a default value of 0.5
    if history_length < 2:
      history_similarity = 0.5
    else:  
      # Calculate the history similarity
      if args.similarity_measure == 'cosine_similarity':
        history_similarity = cosine_similarity(X=history_news_vecs)
        history_similarity = (history_similarity + 1) / 2
      else:
        history_similarity = euclidean_distances(history_news_vecs, history_news_vecs)
        history_similarity = 1 / (1 + history_similarity)
      upper_right = np.triu_indices(history_similarity.shape[0], k=1)
      history_similarity = np.mean(history_similarity[upper_right])
    
    # Calculate the candidate diversities
    if args.similarity_measure == 'cosine_similarity':
      candidate_diversity = (1 - cosine_similarity(X=candidate_news_vecs)) / 2
    else:
      candidate_diversity = euclidean_distances(candidate_news_vecs, candidate_news_vecs)
      candidate_diversity = 1 - (1 / (1 + candidate_diversity))
    candidate_diversity = np.sum(candidate_diversity, axis=1) / (np.shape(candidate_diversity)[1] - 1)
    
    # Softmax the scores
    scores = softmax(scores)
    
    # Override the history similarity score as diversity weight with a given fixed value
    if args.fixed_s > 0.0:
      diversity_weight = args.fixed_s
    else:
      diversity_weight = history_similarity
    
    # Calculate the diverse scores
    if args.reranking_function == 'bound':
      diversity_weight = np.minimum(np.maximum(diversity_weight, args.s_min), args.s_max)
    elif args.reranking_function == 'normalized':
      diversity_weight = (diversity_weight - args.s_min) / (args.s_max - args.s_min)
      candidate_diversity = (candidate_diversity - args.d_min) / (args.d_max - args.d_min)
    diversified_scores = (diversity_weight * candidate_diversity) + ((1 - diversity_weight) * scores) 
    
    # Return the diversified scores, history similarity and candidate diversity
    return diversified_scores, history_similarity, candidate_diversity
    

def user_similarity(args, history_news_vecs, history_log_mask):
    """
    Function for calculating the user similarity.
    Inputs:
        args - Parsed arguments
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
      if args.similarity_measure == 'cosine_similarity':
        history_similarity = cosine_similarity(X=history_news_vecs)
        history_similarity = (history_similarity + 1) / 2
      else:
        history_similarity = euclidean_distances(history_news_vecs, history_news_vecs)
        history_similarity = 1 / (1 + history_similarity)
      upper_right = np.triu_indices(history_similarity.shape[0], k=1)
      history_similarity = np.mean(history_similarity[upper_right])
    
    # Return the history similarity
    return history_similarity