from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
#from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import iterator_utils
import argparse
import copy
import model_helper
import model
import collections
import inference

FLAGS = None

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--data_dir", type=str, default="/home/wangtm/code/RoleLabel/data", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="/S1/LCWM/wangtm/model/", help="Model directory")
    parser.add_argument("--out_dir", type=str, default="/S1/LCWM/wangtm/output/", help="Out directory")
    parser.add_argument("--train_dir", type=str, default="RoleLabel/seq2seq:1/", help="Training directory")
    parser.add_argument("--gpu_device", type=str, default="2", help="which gpu to use")

    parser.add_argument("--train_word_data", type=str, default="train.src1",
                        help="Training data_dst path")
    parser.add_argument("--train_pos_data", type=str, default="train.src2",
                        help="Training data_dst path")
    parser.add_argument("--train_role_data", type=str, default="train.tgt",
                        help="Training data_dst path")

    parser.add_argument("--valid_word_data", type=str, default="valid.src1",
                        help="Training data_dst path")
    parser.add_argument("--valid_pos_data", type=str, default="valid.src2",
                        help="Training data_dst path")
    parser.add_argument("--valid_role_data", type=str, default="valid.tgt",
                        help="Training data_dst path")

    parser.add_argument("--test_word_data", type=str, default="valid.src1",
                        help="Training data_dst path")
    parser.add_argument("--test_pos_data", type=str, default="valid.src2",
                        help="Training data_dst path")
    parser.add_argument("--test_role_data", type=str, default="valid.tgt",
                        help="Training data_dst path")

    parser.add_argument("--word_vocab", type=str, default="word_vocab",
                        help="from vocab path")
    parser.add_argument("--pos_vocab", type=str, default="pos_vocab",
                        help="to vocab path")
    parser.add_argument("--role_vocab", type=str, default="role_vocab",
                        help="to vocab path")

    parser.add_argument("--output_dir", type=str, default="RoleLabel/seq2seq:1/")
    parser.add_argument("--ckpt_dir", type=str, default="RoleLabel/seq2seq:1/",
                        help="model checkpoint directory")


    parser.add_argument("--max_train_data_size", type=int, default=0, help="Limit on the size of training data (0: no limit)")
    parser.add_argument("--attention", type=str, default="", help="""\
          luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
          attention\
          """)
    parser.add_argument("--word_vocab_size", type=int, default=12000, help="NormalWiki vocabulary size")
    parser.add_argument("--pos_vocab_size", type=int, default=32, help="NormalWiki vocabulary size")
    parser.add_argument("--role_vocab_size", type=int, default=8, help="SimpleWiki vocabulary size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--num_units", type=int, default=300, help="Size of each model layer")
    parser.add_argument("--word_emb_dim", type=int, default=200, help="Dimension of word embedding")
    parser.add_argument("--pos_emb_dim", type=int, default=10, help="Dimension of word embedding")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use during training")
    parser.add_argument("--max_gradient_norm", type=float, default=1.0, help="Clip gradients to this norm")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.5, help="Learning rate decays by this much")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_buckets", type=int, default=1, help="Number of buckets")
    parser.add_argument("--src_max_len", type=int, default=300, help="Maximum length of source sentence")
    parser.add_argument("--tgt_max_len", type=int, default=300, help="Maximum length of target sentence")
    parser.add_argument("--input_keep_prob", type=float, default=0.8, help="Dropout input keep prob")
    parser.add_argument("--output_keep_prob", type=float, default=1.0, help="Dropout output keep prob")
    parser.add_argument("--epoch_num", type=int, default=100, help="Number of epoch")


    parser.add_argument("--num_train_epoch", type=int, default=100, help="Number of epoch for training")
    parser.add_argument("--steps_per_eval", type=int, default=2000, help="How many training steps to do per eval/checkpoint")

def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans

def compute_accuracy(predictions, labels):
    ct = 0
    rt = 0
    predictions = predictions.tolist()
    labels = labels.tolist()
    for i in range(len(predictions)):
        p = predictions[i]
        l = labels[i]
        for j in range(len(p)):
            if l[j] != 0:
                if p[j] == l[j]:
                    rt += 1
                ct += 1
    return  rt * 1.0 / ct

def eval(hparams, eval_model, eval_sess):
    with eval_model.graph.as_default():
        loaded_eval_model, global_step = model_helper.create_or_load_model(
            eval_model.model, hparams.ckpt_dir, eval_sess, "eval")
        dev_src1_file = "%s/%s" % (hparams.data_dir, hparams.valid_word_data)
        dev_src2_file = "%s/%s" % (hparams.data_dir, hparams.valid_pos_data)
        dev_tgt_file = "%s/%s" % (hparams.data_dir, hparams.valid_role_data)
        dev_eval_iterator_feed_dict = {
            eval_model.src1_file_placeholder: dev_src1_file,
            eval_model.src2_file_placeholder: dev_src2_file,
            eval_model.tgt_file_placeholder: dev_tgt_file
        }
        eval_sess.run(eval_model.iterator.initializer, feed_dict=dev_eval_iterator_feed_dict)
        count = 0
        total_loss = 0
        total_predict_count = 0
        total_accuracy= 0
        while True:
            try:
                loss, predict_count, batch_size, predictions, labels = loaded_eval_model.eval(eval_sess)
                accuracy = compute_accuracy(predictions, labels)
                total_predict_count += predict_count
                total_loss += loss * batch_size
                total_accuracy += accuracy
                count += 1
            except  tf.errors.OutOfRangeError:
                break
        perplexity = safe_exp(total_loss / total_predict_count)
        accuracy = total_accuracy / (1.0 * count)
        print("  eval: perplexity %.2f   accuracy %.6f" % (perplexity, accuracy))
        return perplexity

def train(hparams, train=True, interact=False):
    model_creator = model.Seq2Seq
    train_model = model_helper.create_train_model(model_creator, hparams)
    eval_model = model_helper.create_eval_model(model_creator, hparams)
    infer_model = model_helper.create_infer_model(model_creator, hparams)

    # dev_src_file = "%s/%s" % (hparams.data_dir, hparams.from_valid_data)
    # dev_tgt_file = "%s/%s" % (hparams.data_dir, hparams.to_valid_data)
    # sample_src_data = inference.load_data(dev_src_file)
    # sample_tgt_data = inference.load_data(dev_tgt_file)

    hparams.add_hparam(name="ckpt_path", value=os.path.join(hparams.ckpt_dir, "seq2seq.ckpt"))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    train_sess = tf.Session(config=config, graph=train_model.graph)
    eval_sess = tf.Session(config=config, graph=eval_model.graph)
    infer_sess = tf.Session(config=config, graph=infer_model.graph)
    with train_model.graph.as_default():
        loaded_train_model, global_step = model_helper.create_or_load_model(
            train_model.model, hparams.ckpt_dir, train_sess, "train")

    skip_count = hparams.batch_size * hparams.epoch_step
    print("# Init train iterator, skipping %d elements" % skip_count)
    train_sess.run(
        train_model.iterator.initializer,
        feed_dict={train_model.skip_count_placeholder: skip_count})
    step_time, ckpt_loss, ckpt_predict_count, ckpt_accuracy = 0.0, 0.0, 0, 0.0
    steps_per_stats, steps_per_eval = 50, 600
    last_stats_step, last_eval_step, last_ppl, llast_ppl = 0, 0, 100000, 1000000
    num_train_steps = hparams.epoch_num * 1000
    while global_step < num_train_steps:
        # ppl = eval(hparams, eval_model, eval_sess)
        # inference.infer(hparams, infer_model, infer_sess)
        start_time = time.time()
        try:
            step_result = loaded_train_model.train(train_sess)
            _, step_loss, predict_count, global_step, predictions, labels = step_result
            accuracy = compute_accuracy(predictions, labels)
            hparams.epoch_step += 1
        except tf.errors.OutOfRangeError:
            hparams.epoch_step = 0
            print("# Finished an epoch, step %d. Perform external evaluation" % global_step)
            inference.infer(hparams, infer_model, infer_sess)
            #TODO eval
            train_sess.run(
                train_model.iterator.initializer,
                feed_dict={train_model.skip_count_placeholder: 0})
            continue

        step_time += (time.time() - start_time)
        ckpt_loss += step_loss * hparams.batch_size
        ckpt_accuracy += accuracy
        ckpt_predict_count += predict_count
        if global_step - last_stats_step >= steps_per_stats:
            avg_accuracy = ckpt_accuracy / (global_step - last_stats_step)
            avg_step_time = step_time / (global_step - last_stats_step)
            ppl = safe_exp(ckpt_loss / ckpt_predict_count)
            last_stats_step = global_step
            print("  global step %d lr %g  step-time %.2fs  ppl %.2f   accuracy %.6f" %
                    (global_step,
                    loaded_train_model.learning_rate.eval(session=train_sess),
                    avg_step_time, ppl, avg_accuracy))
            step_time, ckpt_loss, ckpt_predict_count, ckpt_accuracy = 0.0, 0.0, 0, 0.0

        if global_step - last_eval_step >= steps_per_eval:
            last_eval_step = global_step
            loaded_train_model.saver.save(
                train_sess,
                hparams.ckpt_path,
                global_step=global_step)
            #TODO eval
            ppl = eval(hparams, eval_model, eval_sess)
            # if ppl >= llast_ppl:
            #     lr = loaded_train_model.lr_decay(train_sess)
            #     print(" learning rate decay to %f" % lr)
            llast_ppl = last_ppl
            last_ppl = ppl

    loaded_train_model.saver.save(
        train_sess,
        hparams.ckpt_path,
        global_step=global_step)



def create_hparams(flags):
    return tf.contrib.training.HParams(
        # dir path
        data_dir=flags.data_dir,
        train_dir=flags.train_dir,
        ckpt_dir=flags.ckpt_dir,
        output_dir=flags.output_dir,

        # data params
        batch_size=flags.batch_size,
        word_vocab_size=flags.word_vocab_size,
        pos_vocab_size=flags.pos_vocab_size,
        role_vocab_size=flags.role_vocab_size,

        UNK_ID=data_utils.UNK_ID,
        PAD_ID=data_utils.PAD_ID,
        word_emb_dim=flags.word_emb_dim,
        pos_emb_dim=flags.pos_emb_dim,
        max_train_data_size=flags.max_train_data_size,
        num_train_epoch=flags.num_train_epoch,
        steps_per_eval=flags.steps_per_eval,

        train_word_data=flags.train_word_data,
        train_pos_data=flags.train_pos_data,
        train_role_data=flags.train_role_data,
        valid_word_data=flags.valid_word_data,
        valid_pos_data=flags.valid_pos_data,
        valid_role_data=flags.valid_role_data,
        test_word_data=flags.test_word_data,
        test_pos_data=flags.test_pos_data,
        test_role_data=flags.test_role_data,

        word_vocab=flags.word_vocab,
        pos_vocab=flags.pos_vocab,
        role_vocab=flags.role_vocab,
        share_vocab=True,

        # model params
        input_keep_prob=flags.input_keep_prob,
        output_keep_prob=flags.output_keep_prob,
        init_weight=0.1,
        num_buckets=flags.num_buckets,
        num_units=flags.num_units,
        num_layers=flags.num_layers,
        learning_rate=flags.learning_rate,
        clip_value=flags.max_gradient_norm,
        decay_factor=flags.learning_rate_decay_factor,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,

        #train params
        epoch_num=flags.epoch_num,
        epoch_step=0
    )

def main(_):

    hparams = create_hparams(FLAGS)
    train(hparams)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    FLAGS.ckpt_dir = FLAGS.model_dir + FLAGS.ckpt_dir
    FLAGS.train_dir = FLAGS.model_dir + FLAGS.train_dir
    FLAGS.output_dir = FLAGS.out_dir + FLAGS.output_dir
    print(FLAGS)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device
    tf.app.run()