import tensorflow as tf

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # 쿼리 은닉 상태(query hidden state)는 (batch_size, hidden size)쌍으로 이루어져 있습니다.
        # query_with_time_axis은 (batch_size, 1, hidden size)쌍으로 이루어져 있습니다.
        # values는 (batch_size, max_len, hidden size)쌍으로 이루어져 있습니다.
        # 스코어(score)계산을 위해 덧셈을 수행하고자 시간 축을 확장하여 아래의 과정을 수행합니다.
        query_with_time_axis = tf.expand_dims(query, 1)

        # score는 (batch_size, max_length, 1)쌍으로 이루어져 있습니다.
        # score를 self.V에 적용하기 때문에 마지막 축에 1을 얻습니다.
        # self.V에 적용하기 전에 텐서는 (batch_size, max_length, units)쌍으로 이루어져 있습니다.
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights는 (batch_size, max_length, 1)쌍으로 이루어져 있습니다. 
        attention_weights = tf.nn.softmax(score, axis=1)

        # 덧셈이후 컨텍스트 벡터(context_vector)는 (batch_size, hidden_size)쌍으로 이루어져 있습니다.
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights