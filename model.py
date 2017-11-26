import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core as layers_core

class Seq2Seq():
    def __init__(self, hparams, iterator, mode, src1_vocab_table, src2_vocab_table, tgt_vocab_table):
        self.iterator = iterator
        self.num_layers = hparams.num_layers
        self.src_vocab_size = hparams.src_vocab_size
        self.src_pos_size = hparams.src_pos_size
        self.tgt_class_size = hparams.tgt_vocab_size
        self.word_emb_dim = hparams.word_emb_dim
        self.pos_emb_dim = hparams.pos_emb_dim
        self.num_units = hparams.num_units
        self.learning_rate = tf.Variable(float(hparams.learning_rate), trainable=False)
        self.clip_value = hparams.clip_value
        self.init_weight = hparams.init_weight
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * hparams.decay_factor)
        self.mode = mode
        self.hparams = hparams

        initializer = tf.random_uniform_initializer(-self.init_weight, self.init_weight)
        tf.get_variable_scope().set_initializer(initializer)
        self.batch_size = tf.size(self.iterator.source_sequence_length)

        with tf.variable_scope("embedding") as scope:
            self.word_embeddings = tf.Variable(self.init_matrix([self.src_vocab_size, self.word_emb_dim]))
            self.pos_embeddings = tf.Variable(self.init_matrix([self.src_pos_size, self.pos_emb_dim]))


        with tf.variable_scope("project") as scope:
            self.output_layer = layers_core.Dense(self.tgt_class_size, use_bias=True)

        res = self.build_graph()
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.loss = res[0]
        self.sample = res[2]
        self.alignment = res[3]
        self.global_step = tf.Variable(0, trainable=False)
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.predict_count = tf.reduce_sum(self.iterator.target_sequence_length)


        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            with tf.variable_scope("train_op") as scope:
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                gradients, v = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
                self.train_op = optimizer.apply_gradients(zip(gradients, v), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def _single_cell(self):
        single_cell = tf.contrib.rnn.BasicLSTMCell(self.num_units)
        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell,
                                                   input_keep_prob=self.input_keep_prob,
                                                   output_keep_prob=self.output_keep_prob)
        return single_cell

    def build_graph(self):
        self.input_keep_prob = self.hparams.input_keep_prob
        self.output_keep_prob = self.hparams.output_keep_prob
        with tf.variable_scope("seq2seq"):
            encoder_output, encoder_state = self.build_encoder()

            logits, sample_id, final_context_state, alignment = self.build_decoder(encoder_output, encoder_state)

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                loss = self.compute_loss(logits)
            else:
                loss = "NaN"
        return loss, logits, sample_id, alignment

    def build_encoder(self):
        iterator = self.iterator
        word_emb_inp = tf.nn.embedding_lookup(self.word_embeddings, iterator.source_word)
        pos_emb_inp = tf.nn.embedding_lookup(self.pos_embeddings, iterator.source_pos)
        encoder_emb_inp = tf.concat((word_emb_inp, pos_emb_inp), axis=2)
        if self.num_layers > 1:
            encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([self._single_cell() for _ in range(self.num_layers)])
            encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([self._single_cell() for _ in range(self.num_layers)])
        else:
            encoder_cell_fw = self._single_cell()
            encoder_cell_bw = self._single_cell()

        encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=encoder_cell_fw,
            cell_bw=encoder_cell_bw,
            inputs=encoder_emb_inp,
            dtype=tf.float32,
            sequence_length=iterator.source_sequence_length)

        if self.num_layers > 1:
            encoder_state = []
            for layer_id in range(self.num_layers):
                fw_c, fw_h = bi_encoder_state[0][layer_id]
                bw_c, bw_h = bi_encoder_state[1][layer_id]
                c = (fw_c + bw_c) / 2.0
                h = (fw_h + bw_h) / 2.0
                state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)
                encoder_state.append(state)
            encoder_state = tuple(encoder_state)
        else:
            fw_c, fw_h = bi_encoder_state[0]
            bw_c, bw_h = bi_encoder_state[1]
            c = (fw_c + bw_c) / 2.0
            h = (fw_h + bw_h) / 2.0
            encoder_state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)
        return encoder_outputs, encoder_state

    def build_decoder(self, encoder_outputs, encoder_state):
        iterator = self.iterator
        encoder_outputs_fw, encoder_outputs_bw = encoder_outputs
        memory = tf.concat([encoder_outputs_fw, encoder_outputs_bw], axis=2)
        if self.num_layers > 1:
            decoder_cell = tf.contrib.rnn.MultiRNNCell([self._single_cell() for _ in range(self.num_layers)])
        else:
            decoder_cell = self._single_cell()
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units,
                                                                memory=memory,
                                                                scale=True,
                                                                memory_sequence_length=iterator.source_sequence_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                           attention_mechanism,
                                                           alignment_history=True,
                                                           attention_layer_size=self.num_units)
        initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=encoder_state)
        word_emb_inp = tf.nn.embedding_lookup(self.word_embeddings, iterator.source_word)
        pos_emb_inp = tf.nn.embedding_lookup(self.pos_embeddings, iterator.source_pos)
        decoder_emb_inp = tf.concat((word_emb_inp, pos_emb_inp), axis=2)
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, iterator.target_sequence_length)
        my_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                         helper=helper,
                                                         initial_state=initial_state,
                                                         output_layer=self.output_layer
                                                         )
        decoder_outputs, decoder_state, decoder_output_len = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                                                                   maximum_iterations=100,
                                                                                                   swap_memory=True, )

        sample_id = decoder_outputs.sample_id
        logits = decoder_outputs.rnn_output
        return logits, sample_id, decoder_state, 0


    def get_max_time(self, tensor):
        return tensor.shape[1].value or tf.shape(tensor)[1]

    def compute_loss(self, logits):
        iterator = self.iterator
        max_time = self.get_max_time(iterator.target_output)
        target_weights = tf.sequence_mask(iterator.target_sequence_length, max_time, dtype=tf.float32)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=iterator.target_output, logits=logits)
        loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)
        return loss

    def train(self, sess):
        return sess.run([self.train_op,
                         self.loss,
                         self.predict_count,
                         self.global_step])

    def eval(self, sess):
        return sess.run([self.loss,
                         self.predict_count,
                         self.batch_size])
    def infer(self, sess):
        return sess.run([self.sample,
                         self.alignment])
    def lr_decay(self, sess):
        return sess.run(self.learning_rate_decay_op)

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)