import argparse
from Utils.utils import str2bool


def parse_args():
    """
    Function for parsing the command line arguments.
    """
    
    # Initialize the parser
    parser = argparse.ArgumentParser()
    
    # Loading and saving arguments
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'test', 'train_test'], 
                       help="Mode can be either 'train', 'test' or 'train_test'. Default is train.")
    parser.add_argument("--dataset", type=str, default="MIND", choices=['MIND', 'RTL_NR'], 
                       help="Dataset can be either 'MIND' or 'RTL_NR'. Default is 'MIND'.")
    parser.add_argument("--root_data_dir", type=str, default="./MIND",
                       help="Root directory of the data. Default is './MIND'.")
    parser.add_argument("--train_dir", type=str, default='train',
                       help="Directory of the training data. Default is 'train'.")
    parser.add_argument("--dev_dir", type=str, default='dev',
                       help="Directory of the development/validation data. Default is 'dev'.")
    parser.add_argument("--test_dir", type=str, default='test',
                       help="Directory of the test data. Default is 'test'.")
    parser.add_argument("--model_dir", type=str, default='./model', 
                       help="Directory where the model is saved and loaded from. Default is './model'.")
    parser.add_argument("--load_ckpt_name", type=str, default=None, 
                        help="Checkpoint to load during testing. Default is None (don't load any).")
    parser.add_argument("--save_dev_results", type=str2bool, default=False, 
                       help="Whether to save the test results (recommendations, user similarity and candidate diversity). Default is False.")
    
    # Training arguments
    parser.add_argument("--seed", type=int, default=1234, 
                       help="Seed to use during training. Default is 1234.")
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of epochs to train for. Default is 10.")
    parser.add_argument("--optimizer", type=str, default="adam", choices=['adam', 'adamw', 'sgd', 'sgd_momentum'], 
                       help="Optimizer to use for training. Default is 'adam'.")
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="Learning rate to use during training. Default is 1e-4.")    
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size to use during training and testing. Default is 64.")
    parser.add_argument("--npratio", type=int, default=4, 
                       help="Number of negatives to sample per positive training example. Default is 4.")
    parser.add_argument("--enable_gpu", type=str2bool, default=True, 
                       help="Whether to use the GPU or not. Default is True.")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000,
                       help="Buffer size for shuffling the dataset. Default is 10000.")
    parser.add_argument("--num_workers", type=int, default=0, 
                       help="Number of workers for data loading. Default is 0.")
    parser.add_argument("--log_steps", type=int, default=1000,
                       help="Number of steps before logging intermediate metrics. Default is 1000.")
    parser.add_argument("--patience", type=int, default=3, 
                       help="Number of epochs without improvement in AUC before stopping. Default is 3.")
    parser.add_argument("--filter_num", type=int, default=0, 
                       help="Number of word occurences to filter out. Default is 0 (No filtering).")

    # Model arguments
    parser.add_argument("--architecture", default="plm4newsrec", type=str,
                        choices=['plm4newsrec', 'nrms', 'naml', 'lstur'],
                        help="Which model architecture to use. Default is 'plm4newsrec'.")
    parser.add_argument("--num_unfrozen_layers", type=int, default=2, 
                       help="Number of last layers to keep unfrozen in the encoder model. Default is 2.")
    parser.add_argument("--num_words_title", type=int, default=20, 
                       help="Maximum token length of the title. Default is 20.")
    parser.add_argument("--num_words_abstract", type=int, default=50,
                       help="Maximum token length of the abstract. Default is 50.")
    parser.add_argument("--user_log_length", type=int, default=50,
                       help="Maximum length of the user history. Default is 50.")
    parser.add_argument("--news_encoder_model", default="fastformer", type=str,
                        choices=['bert', 'fastformer'],
                        help="Text encoder model for the news content (only used in plm4newsrec architecture). Default is 'fastformer'.")
    parser.add_argument("--news_dim", type=int, default=64,
                       help="Dimensionality of the news embeddings. Default is 64.")
    parser.add_argument("--news_query_vector_dim", type=int, default=200,
                       help="Dimensionality of the news query vector. Default is 200.")
    parser.add_argument("--user_query_vector_dim", type=int, default=200,
                       help="Dimensionality of the user query vector. ")
    parser.add_argument("--num_attention_heads", type=int, default=20,
                       help="Number of attention heads. Default is 20.")
    parser.add_argument("--drop_rate", type=float, default=0.2,
                       help="Dropout rate to use during training. Default is 0.2.")
    parser.add_argument("--do_lower_case", type=str2bool, default=True, 
                       help="Whether to lower case the data when using NRMS or NAML. Default is True.")
    
    # Diversification arguments
    parser.add_argument("--diversify", type=str2bool, default=False, 
                       help="Whether to diversify the recommendations. Default is False.")
    parser.add_argument("--similarity_measure", type=str, default="cosine_similarity", choices=['cosine_similarity', 'euclidean_distance'], 
                       help="Similarity measure to use for diversification. Only used when diversify=True. Default is 'cosine_similarity'.")
    parser.add_argument("--reranking_function", type=str, default="naive", choices=['naive', 'bound', 'normalized'], 
                       help="Reranking function to use for diversification. Only used when diversify=True. Default is 'naive'.")
    parser.add_argument("--s_min", type=float, default=0.0,
                       help="Minimimum user similarity value. Only used when reranking_function is either 'bound' or 'normalized'. Default is 0.0 (not used).")
    parser.add_argument("--s_max", type=float, default=0.0,
                       help="Maximum user similarity value. Only used when reranking_function is either 'bound' or 'normalized'. Default is 0.0 (not used).")
    parser.add_argument("--d_min", type=float, default=0.0,
                       help="Minimimum candidate diversity value. Only used when reranking_function is 'normalized'. Default is 0.0 (not used).")
    parser.add_argument("--d_max", type=float, default=0.0,
                       help="Maximum candidate diversity value. Only used when reranking_function is 'normalized'. Default is 0.0 (not used).")
    parser.add_argument("--fixed_s", type=float, default=0.0,
                       help="Fixed user similarity weight. Default is 0.0 (weight is determined per user).")
    
    
    # Parse the arguments 
    args = parser.parse_args()
    
    # Return the arguments
    return args