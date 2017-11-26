from __future__ import print_function
import collections
import tensorflow as tf
import data_utils

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]
class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source_word", "source_pos",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
  pass
def get_infer_iterator(src1_dataset,
                       src2_dataset,
                       src1_vocab_table,
                       src2_vocab_table,
                       batch_size,
                       src_max_len=None):
    src_pad_id = data_utils.PAD_ID
    src_dataset = tf.contrib.data.Dataset.zip((src1_dataset, src2_dataset))
    src_dataset = src_dataset.map(lambda src1, src2: (tf.string_split([src1]).values,
                                                      tf.string_split([src2])))

    if src_max_len:
        src_dataset = src_dataset.map(lambda src1 ,src2: (src1[:src_max_len], src2[:src_max_len]))
  # Convert the word strings to ids
    src_dataset = src_dataset.map(
        lambda src1, src2: (tf.cast(src1_vocab_table.lookup(src1), tf.int32),
                            tf.cast(src2_vocab_table.lookup(src2), tf.int32)))
    src_dataset = src_dataset.map(lambda src1, src2: (src1, src2, tf.size(src1)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(
                tf.TensorShape([None]),  # src1
                tf.TensorShape([None]),  # src2
                tf.TensorShape([])),  # src_len
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                src_pad_id,  # src1
                src_pad_id,  # src2
                0))  # src_len -- unused

    batched_dataset = batching_func(src_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src1_ids, src2_ids, src_seq_len) = batched_iter.get_next()
    return BatchedInput(
        initializer=batched_iter.initializer,
        source_word=src1_ids,
        source_pos=src2_ids,
        target_output=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=src_seq_len)

def get_iterator(src1_dataset,
                 src2_dataset,
                 tgt_dataset,
                 src1_vocab_table,
                 src2_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 num_buckets,
                 src_max_len,
                 tgt_max_len,
                 num_threads=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0
                 ):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000
    src_pad_id = data_utils.PAD_ID
    tgt_pad_id = data_utils.PAD_ID

    src_tgt_dataset = tf.contrib.data.Dataset.zip((src1_dataset, src2_dataset, tgt_dataset))
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src1, src2, tgt: (
            tf.string_split([src1]).values, tf.string_split([src2]).values, tf.string_split([tgt]).values),
        num_threads=num_threads,
        output_buffer_size=output_buffer_size)

    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src1, src2, tgt: tf.logical_and(tf.size(src1) > 0, tf.size(src2)>0, tf.size(tgt) > 0))
    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src1, src2, tgt: (src1[:src_max_len], src2[:src_max_len], tgt),
            num_threads=num_threads,
            output_buffer_size=output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src1, src2, tgt: (src1, src2, tgt[:tgt_max_len]),
            num_threads=num_threads,
            output_buffer_size=output_buffer_size)
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src1, src2, tgt: (tf.cast(src1_vocab_table.lookup(src1), tf.int32),
                                 tf.cast(src2_vocab_table.lookup(src2), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_threads=num_threads,
        output_buffer_size=output_buffer_size)
    # src_tgt_dataset = src_tgt_dataset.map(
    #     lambda src1, src2, tgt: (src1, src2,
    #                       tf.concat(([tgt_sos_id], tgt), 0),
    #                       tf.concat((tgt, [tgt_eos_id]), 0)),
    #     num_threads=num_threads,
    #     output_buffer_size=output_buffer_size)
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src1, src2, tgt: (src1, src2, tgt, tf.size(src1)),
        num_threads=num_threads,
        output_buffer_size=output_buffer_size)

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([None]),  # src1
                tf.TensorShape([None]),  # src2
                tf.TensorShape([None]),  # tgt
                tf.TensorShape([])),  # src_tgt_len
                  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                src_pad_id,  # src
                src_pad_id,  # tgt_input
                tgt_pad_id,  # tgt_output
                0))  # src_tgt_len -- unused

    if num_buckets > 1:
        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
            if src_max_len:
                bucket_width = (src_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src1_ids, src2_ids, tgt_output_ids, src_seq_len) = (batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        source_word=src1_ids,
        source_pos=src2_ids,
        target_output=tgt_output_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=src_seq_len)