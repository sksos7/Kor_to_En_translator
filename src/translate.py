import re

from src.plot_attention import plot_attention

from src.evaluate import evaluate

def translate(sentence, init_dict):
    result, sentence, attention_plot = evaluate(sentence, init_dict)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    result = re.sub('<end>', '', result)
    #return result

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

    return result