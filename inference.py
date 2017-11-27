from __future__ import print_function

import codecs
import time

import tensorflow as tf
import model_helper
import data_utils

def load_data(inference_input_file, hparams=None):
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()

  if hparams and hparams.inference_indices:
    inference_data = [inference_data[i] for i in hparams.inference_indices]

  return inference_data

def infer(hparams, infer_model, infer_sess, batch_size=1):
    src1_vocab_file = "%s/%s" % (hparams.data_dir, hparams.word_vocab)
    src2_vocab_file = "%s/%s" % (hparams.data_dir, hparams.pos_vocab)
    tgt_vocab_file = "%s/%s" % (hparams.data_dir, hparams.role_vocab)
    with infer_model.graph.as_default():
        loaded_infer_model, global_step = model_helper.create_or_load_model(
            infer_model.model, hparams.ckpt_dir, infer_sess, "infer")
        test_src1_file = "%s/%s" % (hparams.data_dir, hparams.test_word_data)
        test_src2_file = "%s/%s" % (hparams.data_dir, hparams.test_pos_data)
        test_src1_data = load_data(test_src1_file)
        test_src2_data = load_data(test_src2_file)
        test_tgt_file = "%s/%s_%d" % (hparams.output_dir, hparams.test_role_data, global_step)
        test_infer_iterator_feed_dict = {
            infer_model.src1_placeholder: test_src1_data,
            infer_model.src2_placeholder: test_src2_data,
            infer_model.batch_size_placeholder: batch_size
        }
        print(" start infer..")
        infer_sess.run(infer_model.iterator.initializer, feed_dict=test_infer_iterator_feed_dict)
        outfile = open(test_tgt_file, "w", encoding="utf-8")
        infer_num = 200
        while True and infer_num > 0:
            try:
                predict = loaded_infer_model.infer(infer_sess)
                from_vocab, rev_from_vocab = data_utils.initialize_vocabulary(src1_vocab_file)
                _, rev_to_vocab = data_utils.initialize_vocabulary(tgt_vocab_file)
                sample_output = predict[0].tolist()
                outputs = []
                # alignment = alignment[0].tolist()
                for output in sample_output:
                    outputs.append(tf.compat.as_str(rev_to_vocab[output]))
                #print(" ".join(outputs))
                outfile.write(" ".join(outputs) + "\n")
                infer_num -= 1
            except tf.errors.OutOfRangeError:
                break
        print("  infer done.")
        outfile.close()


