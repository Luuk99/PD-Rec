import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import math


def acc(y_true, y_hat):
    """
    Function for calculating the accuracy.
    Inputs:
        y_true - True labels
        y_hat - Predicted labels
    Outputs:
        accuracy - Accuracy between the predicted and real labels
    """
    
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot

  
def dcg_score(y_true, y_score, k=10):
    """
    Function for calculating the DCG score.
    Inputs:
        y_true - True labels
        y_score - Predicted scores
        k - Top-k items to compare
    Outputs:
        DCG - DCG score for the given K
    """
    
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """
    Function for calculating the nDCG score.
    Inputs:
        y_true - True labels
        y_score - Predicted scores
        k - Top-k items to compare
    Outputs:
        nDCG - nDCG score for the given K
    """
    
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    """
    Function for calculating the MRR score.
    Inputs:
        y_true - True labels
        y_score - Predicted scores
    Outputs:
        MRR - MRR score between the predicted scores and real labels
    """
    
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ctr_score(y_true, y_score, k=1):
    """
    Function for calculating the CTR score.
    Inputs:
        y_true - True labels
        y_score - Predicted scores
        k - Top-k items to compare
    Outputs:
        CTR - CTR score for the given K
    """
    
    return np.mean(y_true)

    
def ILD_score(embeddings, k=10):
    """
    Function for calculating the intra-list distance of the candidates at k.
    Inputs:
        embeddings - Candidate news embeddings
        k - Top k to calculate the ILD for
    Outputs:
        ild - Intra-list distance of the candidates at k
    """
    
    # Take the top k embeddings
    embeddings_at_k = embeddings[:k]
    
    # Calculate the distance matrix
    distance = (1 - cosine_similarity(X=embeddings_at_k)) / 2

    # Get indicies for upper right triangle w/o diagonal
    upper_right = np.triu_indices(distance.shape[0], k=1)

    # Calculate average distance score of all recommended items in list
    ild = np.mean(distance[upper_right])
    
    # Return the intra-list distance
    return ild


def RR_ILD_score(embeddings, scores, labels, k=10):
    """
    Function for calculating the rank and relevance sensitive intra-list distance of the candidates at k.
    Inputs:
        embeddings - Candidate news embeddings
        scores - Scores of the candidate news articles
        labels - Truth labels
        k - Top k to calculate the RR-ILD for
    Outputs:
        ild - Intra-list distance of the candidates at k
    """
    
    def log_rank_discount(k):
        return 1./math.log2(k+2)     
    
    negative_sample = 0.01
    average_distances = []
    average_weights = []
    
    # Take the top k embeddings and labels
    embeddings_at_k = embeddings[:k]
    labels_at_k = labels[:k]
    
    # Calculate the distance matrix
    distance_matrix = (1 - cosine_similarity(X=embeddings_at_k)) / 2 
    
    for i in range(len(labels_at_k)-1):
      distances_i = []
      weights_i = []
      for j in range(i+1, len(labels_at_k)):
        if j == i:
          continue
        
        # Get the distance from the matrix
        distance_ij = distance_matrix[i,j]
        
        # Determine the relevance of j
        relevance_j = 1 if scores[j] == 1 else negative_sample
        
        # Discount based on the rank (assuming that higher ranked items are viewed more)
        rank_discount_j = log_rank_discount(max(0, j-i-1))
        
        distances_i.append(distance_ij * relevance_j * rank_discount_j)
        weights_i.append(relevance_j * rank_discount_j)
      
      # Calculate the average distance of i
      average_distances_i = sum(distances_i)/float(sum(weights_i))
      
      # Determine the relevance of i
      relevance_i = 1 if scores[i] == 1 else negative_sample
      
      # Discount based on the rank (assuming that higher ranked items are viewed more)
      rank_discount_i = log_rank_discount(i)
      
      average_distances.append(average_distances_i * relevance_i * rank_discount_i)
      average_weights.append(rank_discount_i)
    
    # Average the average distances
    rr_ild = sum(average_distances) / float(sum(average_weights))
    
    # Return the rank and relevance sensitive intra-list distance
    return rr_ild