import tensorflow as tf
from src.bahdanauAttention import BahdanauAttention

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # 어텐션을 사용합니다.
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output는 (batch_size, max_length, hidden_size)쌍으로 이루어져 있습니다.
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # 임베딩층을 통과한 후 x는 (batch_size, 1, embedding_dim)쌍으로 이루어져 있습니다.
        x = self.embedding(x)

        # 컨텍스트 벡터와 임베딩 결과를 결합한 이후 x의 형태는 (batch_size, 1, embedding_dim + hidden_size)쌍으로 이루어져 있습니다.
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 위에서 결합된 벡터를 GRU에 전달합니다.
        output, state = self.gru(x)

        # output은 (batch_size * 1, hidden_size)쌍으로 이루어져 있습니다.
        output = tf.reshape(output, (-1, output.shape[2]))

        # output은 (batch_size, vocab)쌍으로 이루어져 있습니다.
        x = self.fc(output)

        return x, state, attention_weights