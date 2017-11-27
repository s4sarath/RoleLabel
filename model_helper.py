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
import tensorflow as tf

import data_utils
import iterator_utils
import argparse
import copy
import collections
from tensorflow.python.ops import lookup_ops

def load_model(model, ckpt, session, name):
  start_time = time.time()
  model.saver.restore(session, ckpt)
  session.run(tf.tables_initializer())
  print(
      "  loaded %s model parameters from %s, time %.2fs" %
      (name, ckpt, time.time() - start_time))
  return model

def create_or_load_model(model, model_dir, session, name):
  """Create translation model and initialize or load parameters in session."""
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print("  created %s model with fresh parameters, time %.2fs" %
                    (name, time.time() - start_time))

  global_step = model.global_step.eval(session=session)
  return model, global_step

def create_vocab_tables(src1_vocab_file, src2_vocab_file, tgt_vocab_file):
  src1_vocab_table = lookup_ops.index_table_from_file(
      src1_vocab_file, default_value=data_utils.UNK_ID)
  src2_vocab_table = lookup_ops.index_table_from_file(
      src2_vocab_file, default_value=data_utils.UNK_ID)

  tgt_vocab_table = lookup_ops.index_table_from_file(
        tgt_vocab_file, default_value=data_utils.UNK_ID)

  return src1_vocab_table, src2_vocab_table, tgt_vocab_table

class TrainModel(
    collections.namedtuple("TrainModel",
                           ("graph", "model", "iterator", "skip_count_placeholder"))):
  pass

def create_train_model(
        model_creator,
        hparams):
    src1_file = "%s/%s" % (hparams.data_dir, hparams.train_word_data)
    src2_file = "%s/%s" % (hparams.data_dir, hparams.train_pos_data)
    tgt_file = "%s/%s" % (hparams.data_dir, hparams.train_role_data)
    src1_vocab_file = "%s/%s" % (hparams.data_dir, hparams.word_vocab)
    src2_vocab_file = "%s/%s" % (hparams.data_dir, hparams.pos_vocab)
    tgt_vocab_file = "%s/%s" % (hparams.data_dir, hparams.role_vocab)

    graph = tf.Graph()
    with graph.as_default(), tf.container("train"):
        src1_vocab_table, src2_vocab_table, tgt_vocab_table = create_vocab_tables(src1_vocab_file, src2_vocab_file, tgt_vocab_file)
        src1_dataset = tf.contrib.data.TextLineDataset(src1_file)
        src2_dataset = tf.contrib.data.TextLineDataset(src2_file)
        tgt_dataset = tf.contrib.data.TextLineDataset(tgt_file)
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)
        iterator = iterator_utils.get_iterator(
            src1_dataset,
            src2_dataset,
            tgt_dataset,
            src1_vocab_table,
            src2_vocab_table,
            tgt_vocab_table,
            hparams.batch_size,
            hparams.num_buckets,
            hparams.src_max_len,
            hparams.tgt_max_len,
            skip_count=skip_count_placeholder)
        with tf.device("/cpu:0"):
            model = model_creator(
                hparams,
                iterator,
                tf.contrib.learn.ModeKeys.TRAIN,
                src1_vocab_table,
                src2_vocab_table,
                tgt_vocab_table)
        return TrainModel(
            graph=graph,
            model=model,
            iterator=iterator,
            skip_count_placeholder=skip_count_placeholder)


class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model", "src1_file_placeholder", "src2_file_placeholder", "tgt_file_placeholder", "iterator"))):
  pass

def create_eval_model(
        model_creator,
        hparams):
    src1_vocab_file = "%s/%s" % (hparams.data_dir, hparams.word_vocab)
    src2_vocab_file = "%s/%s" % (hparams.data_dir, hparams.word_vocab)
    tgt_vocab_file = "%s/%s" % (hparams.data_dir, hparams.role_vocab)

    graph = tf.Graph()
    with graph.as_default(), tf.container("eval"):
        src1_vocab_table, src2_vocab_table, tgt_vocab_table = create_vocab_tables(src1_vocab_file, src2_vocab_file, tgt_vocab_file)
        src1_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        src2_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        src1_dataset = tf.contrib.data.TextLineDataset(src1_file_placeholder)
        src2_dataset = tf.contrib.data.TextLineDataset(src2_file_placeholder)
        tgt_dataset = tf.contrib.data.TextLineDataset(tgt_file_placeholder)
        iterator = iterator_utils.get_iterator(
            src1_dataset,
            src2_dataset,
            tgt_dataset,
            src1_vocab_table,
            src2_vocab_table,
            tgt_vocab_table,
            hparams.batch_size,
            hparams.num_buckets,
            hparams.src_max_len,
            hparams.tgt_max_len)
        with tf.device("/cpu:0"):
            model = model_creator(
                hparams,
                iterator,
                tf.contrib.learn.ModeKeys.EVAL,
                src1_vocab_table,
                src2_vocab_table,
                tgt_vocab_table)
    return EvalModel(
        graph=graph,
        model=model,
        iterator=iterator,
        src1_file_placeholder=src1_file_placeholder,
        src2_file_placeholder=src2_file_placeholder,
        tgt_file_placeholder=tgt_file_placeholder)


class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model", "batch_size_placeholder", "src1_placeholder", "src2_placeholder", "iterator"))):
  pass

def create_infer_model(model_creator, hparams):
    src1_vocab_file = "%s/%s" % (hparams.data_dir, hparams.word_vocab)
    src2_vocab_file = "%s/%s" % (hparams.data_dir, hparams.pos_vocab)
    tgt_vocab_file = "%s/%s" % (hparams.data_dir, hparams.role_vocab)

    graph = tf.Graph()
    with graph.as_default(), tf.container("infer"):
        src1_vocab_table, src2_vocab_table, tgt_vocab_table = create_vocab_tables(src1_vocab_file, src2_vocab_file, tgt_vocab_file)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            tgt_vocab_file, default_value=data_utils._UNK)
        src1_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        src2_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        src1_dataset = tf.contrib.data.Dataset.from_tensor_slices(src1_placeholder)
        src2_dataset = tf.contrib.data.Dataset.from_tensor_slices(src2_placeholder)
        iterator = iterator_utils.get_infer_iterator(
            src1_dataset,
            src2_dataset,
            src1_vocab_table,
            src2_vocab_table,
            batch_size=batch_size_placeholder)
        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.INFER,
            src1_vocab_table=src1_vocab_table,
            src2_vocab_table=src2_vocab_table,
            tgt_vocab_table=tgt_vocab_table,
            )
    return InferModel(
        graph=graph,
        model=model,
        src1_placeholder=src1_placeholder,
        src2_placeholder=src2_placeholder,
        batch_size_placeholder=batch_size_placeholder,
        iterator=iterator)