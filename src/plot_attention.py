import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import mpld3
import io

import json

# 한글 꺠짐 방지
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 어텐션 가중치를 그리기 위한 함수입니다.
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    html_txt = mpld3.fig_to_html(fig, figid='THIS_IS_FIGID')
    html_file = open('templates/image_attention.html', 'w')
    html_file.write(html_txt)
    html_file.close()
    #return img_attention

    plt.savefig('src/attention.png')