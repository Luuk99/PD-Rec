import time
import datetime
import numpy as npR
import torch
from sklearn.metrics import roc_auc_score
from torch.cuda import amp
import os
import torch.optim as optim
import pandas as pd


def order_random(labels, embeddings, news_ids):
    """
    Function for randomly ranking the recommendations.
    Inputs:
        labels - Truth labels for the candidate news articles
        embeddings - Candidate news embeddings
        news_ids - Candidate news ids
    Outputs:
        score - Random scores 
        labels - Ordered truth labels for the candidate news articles
        embeddings - Ordered candidate news embeddings
        news_ids - Ordered candidate news ids
    """
    
    # Generate a random score for each of the candidates
    score = np.random.uniform(0, 1, labels.shape[0])
    
    # Order according to the score
    order = np.argsort(score)[::-1]
    score = np.take(score, order)
    labels = np.take(labels, order)
    embeddings = np.take(embeddings, order, axis=0)
    news_ids = np.take(news_ids, order)
    
    # Return the random score and ordered labels, embeddings and news ids
    return score, labels, embeddings, news_ids
  

def order_diversity(labels, embeddings, news_ids):
    """
    Function for ranking the recommendations according to the diversity.
    Inputs:
        labels - Truth labels for the candidate news articles
        embeddings - Candidate news embeddings
        news_ids - Candidate news ids
    Outputs:
        score - Diversity scores
        labels - Ordered truth labels for the candidate news articles
        embeddings - Ordered candidate news embeddings
        news_ids - Ordered candidate news ids
    """
    
    # Calculate the candidate diversity
    candidate_diversity = (1 - cosine_similarity(X=embeddings, dense_output=False)) /2
    candidate_diversity = np.sum(candidate_diversity, axis=1) / (np.shape(candidate_diversity)[1] - 1)
    
    # Order according to the candidate diversity
    order = np.argsort(candidate_diversity)[::-1]
    score = np.take(candidate_diversity, order)
    labels = np.take(labels, order)
    embeddings = np.take(embeddings, order, axis=0)
    news_ids = np.take(news_ids, order)
    
    # Return the diversity scores and ordered labels, embeddings and news ids
    return score, labels, embeddings, news_ids
  

def perform_baseline_epoch(data_dir, args, word_dict, category_dict, subcategory_dict, dataset='dev'):
    """
    Function for performing a single baseline evaluation epoch.
    Inputs:
        data_dir - Directory containing the data
        args - Parsed arguments
        word_dict - Dictionary containing the words
        category_dict - Dictionary containing the categories
        subcategory_dict - Dictionary containing the subcategories
        dataset - String indicating the partition of the dataset to use
    Outputs:
        (rand_ / div_)AUC - Area Under Curve
        (rand_ / div_)MRR - Mean Reciprocal Rank
        (rand_ / div_)nDCG5 - normalized Discounted Cumulative Gain @ 5
        (rand_ / div_)nDCG10 - normalized Discounted Cumulative Gain @ 10
        (rand_ / div_)ILD5 - Intra-List Distance @ 5
        (rand_ / div_)ILD10 - Intra-List Distance @ 10
        (rand_ / div_)RR_ILD5 - Rank and Relevance sensitive Intra-List Distance @ 5
        (rand_ / div_)RR_ILD10 - Rank and Relevance sensitive Intra-List Distance @ 10
        (rand_ / div_)COV5 - item Coverage @ 5
        (rand_ / div_)COV10 - item Coverage @ 10
        elapsed_time - Total evaluation time
    """
    
    # Initialize the metrics
    rand_AUC = []
    rand_MRR = []
    rand_nDCG5 = []
    rand_nDCG10 = []
    rand_ILD5 = []
    rand_ILD10 = []
    rand_RR_ILD5 = []
    rand_RR_ILD10 = []
    rand_covered_articles5 = set()
    rand_covered_articles10 = set()
    
    div_AUC = []
    div_MRR = []
    div_nDCG5 = []
    div_nDCG10 = []
    div_ILD5 = []
    div_ILD10 = []
    div_RR_ILD5 = []
    div_RR_ILD10 = []
    div_covered_articles5 = set()
    div_covered_articles10 = set()
    
    # Save the recommendataions, user similarity and candidate diversity if given
    rand_results_dict = {'entry_id': [], 'top10_recommendations': [], 'user_sim': [], 'top10_div': []}
    div_results_dict = {'entry_id': [], 'top10_recommendations': [], 'user_sim': [], 'top10_div': [], 'candidate_diversity': []} # DEBUG
    
    # Load the evaluation data
    dataloader, num_articles = load_data(data_dir, args, None, dataset=dataset,
                                         category_dict=category_dict, subcategory_dict=subcategory_dict,
                                         model=None, word_dict=word_dict, baseline=True)
    
    evaluation_time = time.time()
    for cnt, (log_vec, log_mask, news_vec, news_bias, label, news_id, entry_id, log_st_embedding, candidate_st_embedding) in enumerate(dataloader): 
      if label.mean() == 0 or label.mean() == 1:
        continue 
      
      log_vec = log_vec.to(torch.device("cpu")).detach().numpy()
      log_mask = log_mask.to(torch.device("cpu")).detach().numpy()
      
      log_vec = np.squeeze(log_vec, axis=0)
      log_mask = np.squeeze(log_mask, axis=0)
      news_vec = np.squeeze(news_vec, axis=0)
      news_id = np.squeeze(news_id, axis=0)
      label = np.squeeze(label, axis=0)
      entry_id = np.squeeze(entry_id, axis=0)
      
      # Order according to the baselines
      rand_score, rand_label, rand_news_vec, rand_news_id = order_random(label, news_vec, news_id)
      div_score, div_label, div_news_vec, div_news_id = order_diversity(label, news_vec, news_id)
            
      # Calculate the precision metrics
      rand_auc = roc_auc_score(rand_label, rand_score)
      rand_mrr = mrr_score(rand_label, rand_score)
      rand_ndcg5 = ndcg_score(rand_label, rand_score, k=5)
      rand_ndcg10 = ndcg_score(rand_label, rand_score, k=10)
      rand_AUC.append(rand_auc)
      rand_MRR.append(rand_mrr)
      rand_nDCG5.append(rand_ndcg5)
      rand_nDCG10.append(rand_ndcg10)
            
      div_auc = roc_auc_score(div_label, div_score)
      div_mrr = mrr_score(div_label, div_score)
      div_ndcg5 = ndcg_score(div_label, div_score, k=5)
      div_ndcg10 = ndcg_score(div_label, div_score, k=10)
      div_AUC.append(div_auc)
      div_MRR.append(div_mrr)
      div_nDCG5.append(div_ndcg5)
      div_nDCG10.append(div_ndcg10)
            
      # Calculate the diversity metrics
      rand_ild5 = ILD_score(rand_news_vec, k=5)
      rand_ild10 = ILD_score(rand_news_vec, k=10)
      rand_rr_ild5 = RR_ILD_score(rand_news_vec, rand_score, rand_label, k=5)
      rand_rr_ild10 = RR_ILD_score(rand_news_vec, rand_score, rand_label, k=10)
      rand_top_5_ids = rand_news_id[:5]
      rand_top_10_ids = rand_news_id[:10]
      rand_ILD5.append(rand_ild5)
      rand_ILD10.append(rand_ild10)
      rand_RR_ILD5.append(rand_rr_ild5)
      rand_RR_ILD10.append(rand_rr_ild10)
      rand_covered_articles5.update(rand_top_5_ids.tolist())
      rand_covered_articles10.update(rand_top_10_ids.tolist())
            
      div_ild5 = ILD_score(div_news_vec, k=5)
      div_ild10 = ILD_score(div_news_vec, k=10)
      div_rr_ild5 = RR_ILD_score(div_news_vec, div_score, div_label, k=5)
      div_rr_ild10 = RR_ILD_score(div_news_vec, div_score, div_label, k=10)
      div_top_5_ids = div_news_id[:5]
      div_top_10_ids = div_news_id[:10]
      div_ILD5.append(div_ild5)
      div_ILD10.append(div_ild10)
      div_RR_ILD5.append(div_rr_ild5)
      div_RR_ILD10.append(div_rr_ild10)
      div_covered_articles5.update(div_top_5_ids.tolist())
      div_covered_articles10.update(div_top_10_ids.tolist())
            
      # Save the results
      rand_results_dict['entry_id'].append(entry_id)
      rand_results_dict['top10_recommendations'].append(rand_news_id[:10])
      rand_results_dict['user_sim'].append(user_similarity(args, log_vec, log_mask))
      rand_results_dict['top10_div'].append(candidate_diversity(rand_news_vec[:10]))

      div_results_dict['entry_id'].append(entry_id)
      div_results_dict['top10_recommendations'].append(div_news_id[:10])
      div_results_dict['user_sim'].append(user_similarity(args, log_vec, log_mask))
      div_results_dict['top10_div'].append(candidate_diversity(div_news_vec[:10]))
      #DEBUG
      div_results_dict['candidate_diversity'].append(" ".join([str(x) for x in div_score]))
    
    # Calculate the metrics
    rand_AUC = np.array(rand_AUC).mean()
    rand_MRR = np.array(rand_MRR).mean()
    rand_nDCG5 = np.array(rand_nDCG5).mean()
    rand_nDCG10 = np.array(rand_nDCG10).mean()
    rand_ILD5 = np.array(rand_ILD5).mean()
    rand_ILD10 = np.array(rand_ILD10).mean()
    rand_RR_ILD5 = np.array(rand_RR_ILD5).mean()
    rand_RR_ILD10 = np.array(rand_RR_ILD10).mean()
    rand_COV5 = len(rand_covered_articles5) / num_articles
    rand_COV10 = len(rand_covered_articles10) / num_articles
    
    div_AUC = np.array(div_AUC).mean()
    div_MRR = np.array(div_MRR).mean()
    div_nDCG5 = np.array(div_nDCG5).mean()
    div_nDCG10 = np.array(div_nDCG10).mean()
    div_ILD5 = np.array(div_ILD5).mean()
    div_ILD10 = np.array(div_ILD10).mean()
    div_RR_ILD5 = np.array(div_RR_ILD5).mean()
    div_RR_ILD10 = np.array(div_RR_ILD10).mean()
    div_COV5 = len(div_covered_articles5) / num_articles
    div_COV10 = len(div_covered_articles10) / num_articles
    
    elapsed_time = str(datetime.timedelta(seconds=time.time() - evaluation_time))
    
    # Save the evaluation results
    df = pd.DataFrame.from_dict(rand_results_dict)
    df = df.sort_values('entry_id')
    results_path = os.path.join(args.model_dir, 'random_test_results.csv')
    df.to_csv(results_path)
    print('Random evaluation results saved to: {}'.format(results_path))
    df = pd.DataFrame.from_dict(div_results_dict)
    df = df.sort_values('entry_id')
    results_path = os.path.join(args.model_dir, 'diversity_test_results.csv')
    df.to_csv(results_path)
    print('Diversity evaluation results saved to: {}'.format(results_path))
    
    # Return the metrics
    return rand_AUC, rand_MRR, rand_nDCG5, rand_nDCG10, rand_ILD5, rand_ILD10, rand_RR_ILD5, rand_RR_ILD10, rand_COV5, rand_COV10, div_AUC, div_MRR, div_nDCG5, div_nDCG10, div_ILD5, div_ILD10, div_RR_ILD5, div_RR_ILD10, div_COV5, div_COV10, elapsed_time
  
  
def test_baselines(args, data_dir):
    """
    Function for performing testing.
    Inputs:
        args - Parsed arguments
        data_temp_dir - Temporary directory containing the data
    """
    
    # Load the training data
    _, category_dict, subcategory_dict, word_dict = load_data(data_dir, args, None, dataset='train')
    
    # Perform a test epoch
    print('Testing..')
    rand_AUC, rand_MRR, rand_nDCG5, rand_nDCG10, rand_ILD5, rand_ILD10, rand_RR_ILD5, rand_RR_ILD10, rand_COV5, rand_COV10, div_AUC, div_MRR, div_nDCG5, div_nDCG10, div_ILD5, div_ILD10, div_RR_ILD5, div_RR_ILD10, div_COV5, div_COV10, elapsed_time = perform_baseline_epoch(data_dir, args, word_dict, category_dict, subcategory_dict, dataset='test')
    print('Testing finished')
    print('rand_AUC = {:.5f} & rand_MRR = {:.5f} & rand_nDCG@5 = {:.5f} & rand_nDCG@10 = {:.5f} & rand_ILD@5 = {:.5f} & rand_ILD@10 = {:.5f} & rand_RR-ILD@5 = {:.5f} & rand_RR-ILD@10 = {:.5f} & rand_COV@5 = {:.5f} & rand_COV@10 = {:.5f}'.format(rand_AUC, rand_MRR, rand_nDCG5, rand_nDCG10, rand_ILD5, rand_ILD10, rand_RR_ILD5, rand_RR_ILD10, rand_COV5, rand_COV10))
    print('div_AUC = {:.5f} & div_MRR = {:.5f} & div_nDCG@5 = {:.5f} & div_nDCG@10 = {:.5f} & div_ILD@5 = {:.5f} & div_ILD@10 = {:.5f} & div_RR-ILD@5 = {:.5f} & div_RR-ILD@10 = {:.5f} & div_COV@5 = {:.5f} & div_COV@10 = {:.5f}'.format(div_AUC, div_MRR, div_nDCG5, div_nDCG10, div_ILD5, div_ILD10, div_RR_ILD5, div_RR_ILD10, div_COV5, div_COV10))
    
    # Save the metrics
    print('Saving metrics..')
    metric_dict = {}
    metric_dict[0] = [rand_AUC, rand_MRR, rand_nDCG5, rand_nDCG10, rand_ILD5, rand_ILD10, rand_RR_ILD5, rand_RR_ILD10, rand_COV5, rand_COV10, elapsed_time]
    metrics_df = pd.DataFrame.from_dict(metric_dict, orient='index', columns=['test_AUC',
                                                                            'test_MRR', 'test_nDCG5', 'test_nDCG10',
                                                                            'test_ILD5', 'test_ILD10', 'test_RR_ILD5',
                                                                            'test_RR_ILD10', 'test_COV5', 'test_COV10',
                                                                            'test_duration'])
    metrics_df.index.name = 'epoch'
    metrics_path = os.path.join(args.model_dir, 'rand_test_metrics.csv')
    metrics_df.to_csv(metrics_path)
    print('Random test metrics saved to: {}'.format(metrics_path))
    
    metric_dict = {}
    metric_dict[0] = [div_AUC, div_MRR, div_nDCG5, div_nDCG10, div_ILD5, div_ILD10, div_RR_ILD5, div_RR_ILD10, div_COV5, div_COV10, elapsed_time]
    metrics_df = pd.DataFrame.from_dict(metric_dict, orient='index', columns=['test_AUC',
                                                                            'test_MRR', 'test_nDCG5', 'test_nDCG10',
                                                                            'test_ILD5', 'test_ILD10', 'test_RR_ILD5',
                                                                            'test_RR_ILD10', 'test_COV5', 'test_COV10',
                                                                            'test_duration'])
    metrics_df.index.name = 'epoch'
    metrics_path = os.path.join(args.model_dir, 'div_test_metrics.csv')
    metrics_df.to_csv(metrics_path)
    print('Diverse test metrics saved to: {}'.format(metrics_path))

    
if __name__ == "__main__":
    args = parse_args()
    test_baselines(args=args, data_dir=temp_dir)