'''
超参数配置
'''
from utils.symbols import graphemes_char2id, phonemes_char2id

class Hyperparameters:
    # ###############################################
    #                  Data
    # ###############################################
    device = "cpu"
    data_dir = './data/'
    train_dataset_path = './data/data_train.json'    
    val_dataset_path = './data/data_val.json'    
    test_dataset_path = './data/data_test.json'    
    seed = 123
    # ###############################################
    #                  Model
    # ###############################################
    # encoder
    encoder_layer = 6
    encoder_dim = 512
    encoder_drop_prob = 0.1
    grapheme_size = len(graphemes_char2id)
    encoder_max_input = 30  # 数据集中 最长是 28
    # multi head 
    nhead = 8

    # FFN  比较耗时  linear layer -> conv1d -> conv1d+DS
    encoder_feed_forward_dim = 2*encoder_dim
    decoder_feed_forward_dim = 2*encoder_dim
    ffn_drop_prob = 0.3

    # decoder
    decoder_layer = 6
    decoder_dim = encoder_dim
    decoder_drop_prob = 0.1
    phoneme_size = len(phonemes_char2id)
    MAX_DECODE_STEP = 50

    ENCODER_SOS_IDX = graphemes_char2id['<s>']
    ENCODER_EOS_IDX = graphemes_char2id['</s>']
    ENCODER_PAD_IDX = graphemes_char2id['<pad>']

    DECODER_SOS_IDX = phonemes_char2id['<s>']
    DECODER_EOS_IDX = phonemes_char2id['</s>']
    DECODER_PAD_IDX = phonemes_char2id['<pad>']

    # ###############################################
    #                  Experiment
    # ###############################################
    bs = 128
    init_lr = 1e-4
    epochs = 100
    verbose_step = 100
    save_step = 500
    gard_clip_thresh = 1.

HP = Hyperparameters()

