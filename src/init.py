import os
import tensorflow as tf

from src.processing import load_dataset
from src.decoder import Decoder
from src.encoder import Encoder


def init():
    path_to_file = "src/kor.txt"

    num_examples = 3700
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

    BATCH_SIZE = 64
    embedding_dim = 256
    units = 1024
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)
    print(vocab_inp_size, vocab_tar_size)

    optimizer = tf.keras.optimizers.Adam()

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    checkpoint_dir = 'src/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)

    init_dict ={
        'path_to_file' : "src/kor.txt",
        'num_examples' : 3700,
        'input_tensor' : input_tensor,
        'target_tensor' : target_tensor,
        'inp_lang' : inp_lang,
        'targ_lang' : targ_lang,
        'max_length_targ' : max_length_targ,
        'max_length_inp' : max_length_inp,
        'BATCH_SIZE' : BATCH_SIZE,
        'embedding_dim' : embedding_dim,
        'units' : units,
        'vocab_inp_size' : vocab_inp_size,
        'vocab_tar_size' : vocab_tar_size,
        'optimizer' : optimizer,
        'encoder' : encoder,
        'decoder' : decoder,
        'checkpoint_dir' : checkpoint_dir,
        'checkpoint_prefix' : checkpoint_prefix,
        'checkpoint' : checkpoint
    }

    return init_dict