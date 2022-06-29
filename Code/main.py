import time
import datetime
import numpy as npR
import torch
from sklearn.metrics import roc_auc_score
from torch.cuda import amp
import os
import torch.optim as optim
import pandas as pd

from utils import initialize_tokenizer, initialize_model
from parameters import parse_args
from dataloading import load_data


def perform_training_epoch(model, optimizer, scaler, dataloader, args):
    """
    Function for performing a single training epoch.
    Inputs:
        model - Model to perform the training epoch on
        optimizer - Optimizer to train the model with
        dataloader - Dataloader containing the training data
        args - Parsed arguments
    Outputs:
        loss - Average loss over the entire epoch
        accuracy - Average accuracy over the enitre epoch
        elapsed_time - Execution time for a single training epoch
    """
    
    # Set model to training 
    model.train()
    torch.set_grad_enabled(True)
    
    loss = 0.0
    accuracy = 0.0
    start_time = time.time()
    epoch_time = time.time()
    for cnt, (log_ids, log_mask, input_ids, targets, user_ids) in enumerate(dataloader):
      optimizer.zero_grad()
      
      if args.enable_gpu:
        log_ids = log_ids.cuda(non_blocking=True)
        log_mask = log_mask.cuda(non_blocking=True)
        input_ids = input_ids.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        user_ids = user_ids.cuda(non_blocking=True)
      
      # Pass through the model
      with amp.autocast():
        if args.architecture == 'lstur':
          bz_loss, y_hat = model(input_ids, log_ids, log_mask, user_ids, targets)
        else:
          bz_loss, y_hat = model(input_ids, log_ids, log_mask, targets)
      
      # Optimize the model
      if args.enable_gpu:
        scaler.scale(bz_loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        bz_loss.backward()
        optimizer.step()
      loss += bz_loss.data.float()
      accuracy += acc(targets, y_hat)
      
      # Log the metrics for every given period
      if cnt % args.log_steps == 0:
        end_time = time.time()
        elapsed_time = str(datetime.timedelta(seconds=end_time - start_time))
        start_time = end_time
        print('Ed: {}, train_loss: {:.5f}, acc: {:.5f}, time_elapsed: {}'.format(
          cnt * args.batch_size, loss.data / cnt, accuracy / cnt, elapsed_time))
      
      # Break if the loss is NaN
      if torch.isnan(bz_loss).any():
        return None, None, None
      
    # Calculate the loss, accuracy and elapsed time
    loss /= cnt
    accuracy /= cnt
    elapsed_time = str(datetime.timedelta(seconds=time.time() - epoch_time))
      
    # Return the loss, accuracy and time elapsed
    return loss, accuracy, elapsed_time
  

def perform_evaluation_epoch(model, tokenizer, word_dict, data_dir, args, category_dict, subcategory_dict, dataset='dev'):
    """
    Function for performing a single evaluation epoch.
    Inputs:
        model - Model to perform the training epoch on
        tokenizer - News encoder tokenizer instance
        word_dict - Dictionary containing the words
        data_dir - Directory containing the data
        args - Parsed arguments
        category_dict - Dictionary containing the categories
        subcategory_dict - Dictionary containing the subcategories
        dataset - Dataset to use for performing evaluation (dev or test)
    Outputs:
        AUC - Area Under Curve
        MRR - Mean Reciprocal Rank
        nDCG5 - normalized Discounted Cumulative Gain @ 5
        nDCG10 - normalized Discounted Cumulative Gain @ 10
        ILD5 - Intra-List Distance @ 5
        ILD10 - Intra-List Distance @ 10
        RR_ILD5 - Rank and Relevance sensitive Intra-List Distance @ 5
        RR_ILD10 - Rank and Relevance sensitive Intra-List Distance @ 10
        COV5 - item Coverage @ 5
        COV10 - item Coverage @ 10
        elapsed_time - Total evaluation time
    """
    
    # Set model to evaluation
    model.eval()
    torch.set_grad_enabled(False)
    
    # Initialize the metrics
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    ILD5 = []
    ILD10 = []
    RR_ILD5 = []
    RR_ILD10 = []
    covered_articles5 = set()
    covered_articles10 = set()
    
    # Save the recommendataions, user similarity and candidate diversity if given
    if args.save_dev_results:
      dev_results_dict = {'entry_id': [], 'top10_recommendations': [], 'user_sim': [], 'ild@10': [], 'ild@5': [], 'mrr': [], 'ndcg@10': []}
    
    # Load the evaluation data
    dataloader, num_articles = load_data(data_dir, args, tokenizer, dataset=dataset,
                                         category_dict=category_dict, subcategory_dict=subcategory_dict,
                                         model=model, word_dict=word_dict)
    
    # Iterate over the dataloader
    evaluation_time = time.time()
    for cnt, (log_vecs, log_masks, news_vecs, news_bias, labels, news_ids, entry_ids, log_st_embeddings, candidate_st_embeddings, user_ids) in enumerate(dataloader):
        if args.enable_gpu:
            log_vecs = log_vecs.cuda(non_blocking=True)
            log_masks = log_masks.cuda(non_blocking=True)
            user_ids = user_ids.cuda(non_blocking=True)
        
        # Encode the user embeddings
        if args.architecture == 'lstur':
          users = model.user_embedding(user_ids.long())
          user_vecs = model.user_encoder(users, log_vecs, log_masks).to(torch.device("cpu")).detach().numpy()
        else:
          user_vecs = model.user_encoder(log_vecs, log_masks).to(torch.device("cpu")).detach().numpy()
        log_vecs = log_vecs.to(torch.device("cpu")).detach().numpy()
        log_masks = log_masks.to(torch.device("cpu")).detach().numpy()
        
        # Iterate over item in the batch
        for index, user_vec, log_vec, log_mask, news_vec, bias, label, news_id, entry_id, log_st_embedding, candidate_st_embedding in zip(range(len(labels)), user_vecs, log_vecs, log_masks, news_vecs, news_bias, labels, news_ids, entry_ids, log_st_embeddings, candidate_st_embeddings):
                
            if label.mean() == 0 or label.mean() == 1:
                continue
            
            # Score the candidates
            score = np.dot(news_vec, user_vec)
            
            # Check whether to diversify
            if args.diversify:
              score, user_sim, candidate_div = diversity_reranking(args, log_st_embedding, log_mask, candidate_st_embedding, score)
            
            # Rank the recommendations
            score, label, candidate_st_embedding, news_id = rank_recommendations(score, label, candidate_st_embedding, news_id)
            
            # Calculate the precision metrics
            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)
            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)
            
            # Calculate the diversity metrics
            ild5 = ILD_score(candidate_st_embedding, k=5)
            ild10 = ILD_score(candidate_st_embedding, k=10)
            rr_ild5 = RR_ILD_score(candidate_st_embedding, score, label, k=5)
            rr_ild10 = RR_ILD_score(candidate_st_embedding, score, label, k=10)
            top_5_ids = news_id[:5]
            top_10_ids = news_id[:10]
            ILD5.append(ild5)
            ILD10.append(ild10)
            RR_ILD5.append(rr_ild5)
            RR_ILD10.append(rr_ild10)
            covered_articles5.update(top_5_ids.tolist())
            covered_articles10.update(top_10_ids.tolist())
            
            # Save the results if given
            if args.save_dev_results:
              dev_results_dict['entry_id'].append(entry_id)
              dev_results_dict['top10_recommendations'].append(' '.join(news_id[:10]))
              dev_results_dict['user_sim'].append(user_similarity(args, log_st_embedding, log_mask))
              dev_results_dict['ild@10'].append(ild10)
              dev_results_dict['ild@5'].append(ild5)
              dev_results_dict['mrr'].append(mrr)
              dev_results_dict['ndcg@10'].append(ndcg10)
    
    # Calculate the metrics
    AUC = np.array(AUC).mean()
    MRR = np.array(MRR).mean()
    nDCG5 = np.array(nDCG5).mean()
    nDCG10 = np.array(nDCG10).mean()
    ILD5 = np.array(ILD5).mean()
    ILD10 = np.array(ILD10).mean()
    RR_ILD5 = np.array(RR_ILD5).mean()
    RR_ILD10 = np.array(RR_ILD10).mean()
    COV5 = len(covered_articles5) / num_articles
    COV10 = len(covered_articles10) / num_articles
    elapsed_time = str(datetime.timedelta(seconds=time.time() - evaluation_time))
    
    # Save the evaluation results if given
    if args.save_dev_results:
      df = pd.DataFrame.from_dict(dev_results_dict)
      df = df.sort_values('entry_id')
      results_path = os.path.join(args.model_dir, dataset + '_evaluation_results.csv')
      df.to_csv(results_path)
      print('Evaluation results saved to: {}'.format(results_path))
    
    # Return the metrics
    return AUC, MRR, nDCG5, nDCG10, ILD5, ILD10, RR_ILD5, RR_ILD10, COV5, COV10, elapsed_time
  

def train(args, data_dir):
    """
    Function for performing training.
    Inputs:
        args - Parsed arguments
        data_dir - Directory containing the data
    Outputs:
        model - Trained model
        tokenizer - Tokenizer of the model
        word_dict - Word dictionary
        category_dict - Category dictionary
        subcategory_dict - Subcategory dictionary
    """
    
    # Set a seed
    if args.enable_gpu:
      torch.manual_seed(args.seed)
    
    # Load the tokenizer if needed
    if args.architecture == 'plm4newsrec' and args.news_encoder_model != 'fastformer':
      tokenizer = initialize_tokenizer(args)
    else:
      tokenizer = None
    
    # Load the training data
    train_dataloader, category_dict, subcategory_dict, word_dict, user_id_dict = load_data(data_dir, args, tokenizer, dataset='train')
    
    # Load the recommender model
    model = initialize_model(args, word_dict, category_dict, subcategory_dict, user_id_dict)
    if args.enable_gpu:
        model = model.cuda()
    
    # Initialize the optimizer
    if args.optimizer == 'adam':
      optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
      optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
      optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
      optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Initialize the gradient scaler
    scaler = amp.GradScaler()
    
    # Train the model
    metric_dict = {}
    epochs_without_improvement = 0
    best_AUC = 0
    print('Training...')
    for epoch in range(1, args.epochs + 1):
        # Perform a training epoch
        loss, accuracy, epoch_time = perform_training_epoch(model, optimizer, scaler, train_dataloader, args)
        
        # Break if training was aborted due to NaN loss
        if loss is None:
          print('Breaking due to NaN loss')
          break
        
        # Print training metrics
        print('Training epoch {} finished: loss = {:.5f} & acc = {:.5f} & time_elapsed = {}'.format(epoch, loss,
                                                                                                    accuracy,
                                                                                                    epoch_time))
        
        # Perform an evaluation epoch
        AUC, MRR, nDCG5, nDCG10, ILD5, ILD10, RR_ILD5, RR_ILD10, COV5, COV10, eval_time = perform_evaluation_epoch(model, tokenizer, word_dict, data_dir, args, category_dict, subcategory_dict, dataset='dev')
        print('Evaluation epoch {} finished: AUC = {:.5f} & MRR = {:.5f} & nDCG@5 = {:.5f} & nDCG@10 = {:.5f} & ILD@5 = {:.5f} & ILD@10 = {:.5f} & RR-ILD@5 = {:.5f} & RR-ILD@10 = {:.5f} & COV@5 = {:.5f} & COV@10 = {:.5f}'.format(epoch, AUC, MRR, nDCG5, nDCG10, ILD5, ILD10, RR_ILD5, RR_ILD10, COV5, COV10))
        
        # Save the metrics
        metric_dict[epoch] = [loss.item(), accuracy.item(), epoch_time, AUC, MRR, nDCG5, nDCG10, ILD5, ILD10,
                              RR_ILD5, RR_ILD10, COV5, COV10, eval_time]
        
        # Save the metrics
        print('Saving metrics..')
        metrics_df = pd.DataFrame.from_dict(metric_dict, orient='index', columns=['train_loss', 'train_acc',
                                                                            'train_duration', 'dev_AUC',
                                                                            'dev_MRR', 'dev_nDCG5', 'dev_nDCG10',
                                                                            'dev_ILD5', 'dev_ILD10', 'dev_RR_ILD5',
                                                                            'dev_RR_ILD10', 'dev_COV5', 'dev_COV10',
                                                                            'dev_duration'])
        metrics_df.index.name = 'epoch'
        metrics_path = os.path.join(args.model_dir, 'metrics_epoch{}.csv'.format(epoch))
        metrics_df.to_csv(metrics_path)
        print('Metrics saved to: {}'.format(metrics_path))
        
        # Save the model if it's better
        if AUC > best_AUC:
          # Save the model
          ckpt_path = os.path.join(args.model_dir, 'model-checkpoint.pt')
          torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'category_dict': category_dict,
            'word_dict': word_dict,
            'subcategory_dict': subcategory_dict
          }, ckpt_path)
          print(f"Model saved to {ckpt_path}")
        
        # Check if early stopping has been met 
        if AUC - best_AUC > 0.01: # We use 0.01 to take small changes into account
          best_AUC = AUC
          epochs_without_improvement = 0
        else:
          epochs_without_improvement += 1
          if epochs_without_improvement == args.patience:
            print('Early stopping criteria reached')
            break
    
    # Training is finished
    print('Training finished')
    
    # Save the metrics
    print('Saving metrics..')
    metrics_df = pd.DataFrame.from_dict(metric_dict, orient='index', columns=['train_loss', 'train_acc',
                                                                            'train_duration', 'dev_AUC',
                                                                            'dev_MRR', 'dev_nDCG5', 'dev_nDCG10',
                                                                            'dev_ILD5', 'dev_ILD10', 'dev_RR_ILD5',
                                                                            'dev_RR_ILD10', 'dev_COV5', 'dev_COV10',
                                                                            'dev_duration'])
    metrics_df.index.name = 'epoch'
    metrics_path = os.path.join(args.model_dir, 'training_metrics.csv')
    metrics_df.to_csv(metrics_path)
    print('Metrics saved to: {}'.format(metrics_path))
    
    # Return the model, tokenizer, word_dict, category_dict and subcategory_dict
    return model, tokenizer, word_dict, category_dict, subcategory_dict

  
def test(model, tokenizer, word_dict, category_dict, subcategory_dict, args, data_dir):
    """
    Function for performing testing.
    Inputs:
        model - Trained model to test
        tokenizer - Tokenizer of the model
        word_dict - Word dictionary
        category_dict - Category dictionary
        subcategory_dict - Subcategory dictionary
        args - Parsed arguments
        data_dir - Directory containing the data
    """
    
    # Perform an evaluation epoch
    AUC, MRR, nDCG5, nDCG10, ILD5, ILD10, RR_ILD5, RR_ILD10, COV5, COV10, eval_time = perform_evaluation_epoch(model, tokenizer, word_dict, data_dir, args, category_dict, subcategory_dict, dataset='test')
    print('Testing finished: AUC = {:.5f} & MRR = {:.5f} & nDCG@5 = {:.5f} & nDCG@10 = {:.5f} & ILD@5 = {:.5f} & ILD@10 = {:.5f} & RR-ILD@5 = {:.5f} & RR-ILD@10 = {:.5f} & COV@5 = {:.5f} & COV@10 = {:.5f}'.format(AUC, MRR, nDCG5, nDCG10, ILD5, ILD10, RR_ILD5, RR_ILD10, COV5, COV10))
    
    # Save the metrics
    metric_dict = {}
    metric_dict[0] = [AUC, MRR, nDCG5, nDCG10, ILD5, ILD10, RR_ILD5, RR_ILD10, COV5, COV10, eval_time]
    print('Saving metrics..')
    metrics_df = pd.DataFrame.from_dict(metric_dict, orient='index', columns=['test_AUC',
                                                                            'test_MRR', 'test_nDCG5', 'test_nDCG10',
                                                                            'test_ILD5', 'test_ILD10', 'test_RR_ILD5',
                                                                            'test_RR_ILD10', 'test_COV5', 'test_COV10',
                                                                            'test_duration'])
    metrics_df.index.name = 'epoch'
    metrics_path = os.path.join(args.model_dir, 'test_metrics.csv')
    metrics_df.to_csv(metrics_path)
    print('Metrics saved to: {}'.format(metrics_path))


def load_from_checkpoint(args):
    """
    Function for loading a model from a checkpoint.
    Inputs:
        args - Parsed arguments
    Outputs:
        model - Loaded model
        tokenizer - Tokenizer of the model
        word_dict - Loaded word dictionary
        category_dict - Loaded category dictionary
        subcategory_dict - Loaded subcategory dictionary
    """
    
    # Load the checkpoint
    assert args.load_ckpt_name is not None, 'No checkpoint name provided'
    ckpt_path = get_checkpoint(args.model_dir, args.load_ckpt_name)
    assert ckpt_path is not None, 'No checkpoint found'
    checkpoint = torch.load(ckpt_path)
    
    # Load the category, word, and subcategory dictionaries
    category_dict = checkpoint['category_dict']
    word_dict = checkpoint['word_dict']
    subcategory_dict = checkpoint['subcategory_dict']
    
    # Load the tokenizer if needed
    if args.architecture == 'plm4newsrec' and args.news_encoder_model != 'fastformer':
      tokenizer = initialize_tokenizer(args)
    else:
      tokenizer = None
    
    # Load the recommender model
    model = initialize_model(args, word_dict, category_dict, subcategory_dict)
    if args.enable_gpu:
        model = model.cuda()
    
    # Load the model from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded from {}".format(ckpt_path))
    
    # Return the model, tokenizer, word_dict, category_dict and subcategory_dict
    return model, tokenizer, word_dict, category_dict, subcategory_dict 

    
def main(args):
    """
    Function for starting training and/or testing.
    Inputs:
        args - Parsed arguments
    """
    
    if args.mode == 'train':
      # Train the model
      _, _, _, _, _ = train(args=args, data_dir=args.root_data_dir)
    elif args.mode == 'train_test':
      # Train the model
      model, tokenizer, word_dict, category_dict, subcategory_dict = train(args=args, data_dir=args.root_data_dir)
      # Test the model
      test(model, tokenizer, word_dict, category_dict, subcategory_dict, args=args, data_dir=args.root_data_dir)
    else:
      # Load the model
      model, tokenizer, word_dict, category_dict, subcategory_dict = load_from_checkpoint(args)
      # Test the model
      test(model, tokenizer, word_dict, category_dict, subcategory_dict, args=args, data_dir=args.root_data_dir)


if __name__ == "__main__":
  args = parse_args()
  main(args)