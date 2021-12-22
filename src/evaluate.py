import tensorflow as tf
import numpy as np

from src.preprocess_sentence import preprocess_sentence_kr

def evaluate(sentence, init_dict):
    max_length_targ = init_dict['max_length_targ']
    max_length_inp = init_dict['max_length_inp']

    inp_lang = init_dict['inp_lang']
    targ_lang = init_dict['targ_lang']
    
    encoder = init_dict['encoder']
    decoder = init_dict['decoder']
    units = init_dict['units']

    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence_kr(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                            maxlen=max_length_inp,
                                                            padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                dec_hidden,
                                                                enc_out)

        # 나중에 어텐션 가중치를 시각화하기 위해 어텐션 가중치를 저장합니다.
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # 예측된 ID를 모델에 다시 피드합니다.
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot