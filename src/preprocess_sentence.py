import re

def preprocess_sentence(w):
    w = w.lower().strip()

    # 단어와 단어 뒤에 오는 구두점(.)사이에 공백을 생성합니다.
    # 예시: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # (a-z, A-Z, ".", "?", "!", ",")을 제외한 모든 것을 공백으로 대체합니다.
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # 문장에 start와 end 토큰을 추가합니다.
    w = '<start> ' + w + ' <end>'
    return w

def preprocess_sentence_kr(w):
    w = w.strip()

    # 단어와 단어 뒤에 오는 구두점(.)사이에 공백을 생성합니다.
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    w = re.sub(r'[ |ㄱ-ㅎ|ㅏ-ㅣ]+', " ", w)

    w = w.strip()

    # 문장에 start와 end 토큰을 추가합니다.
    w = '<start> ' + w + ' <end>'
    return w