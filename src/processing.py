import pandas as pd
import tensorflow as tf

from src.preprocess_sentence import preprocess_sentence
from src.preprocess_sentence import preprocess_sentence_kr

def create_dataset(path, num_examples):
  data = pd.read_csv(path, delimiter = "\t")
  data.columns = ["en", "kor", "cc"]
  en = [preprocess_sentence(l) for l in data['en'][:num_examples]]
  kr = [preprocess_sentence_kr(l) for l in data['kor'][:num_examples]]

  # 다음과 같은 형식으로 문장의 쌍을 반환합니다: [영어, 한국어]
  return en, kr # 이렇게 하면 한->영 번역이 됨

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
  # 전처리된 타겟 문장과 입력 문장 쌍을 생성합니다.
  targ_lang, inp_lang = create_dataset(path, num_examples)

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer