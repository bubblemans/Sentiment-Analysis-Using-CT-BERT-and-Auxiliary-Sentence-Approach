# -*- coding: utf-8 -*-
import pandas as pd
import string
import logging
import time

from collections import defaultdict

import sklearn
import torch
import numpy as np

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.model_selection import KFold

logging.basicConfig(filename='./pair-result',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


def f1_score_none(y_true, y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred, average='weighted')


def precision_score_none(y_true, y_pred):
    return sklearn.metrics.precision_score(y_true, y_pred, average='weighted')


def recall_score_none(y_true, y_pred):
    return sklearn.metrics.recall_score(y_true, y_pred, average='weighted')


def questionAndAnswer(model_type, pretrained_model, model_args, qa_data):
    logging.info('epochs: {}, learning_rate: {}'.format(model_args.num_train_epochs, model_args.learning_rate))
    logging.info('train_batch_size: {}, eval_batch_size: {}'.format(model_args.train_batch_size, model_args.eval_batch_size))
    logging.info('sequence length: {}'.format(model_args.max_seq_length))

    start = time.time()
    num_folds = 5
    nth_fold = 0
    kf = KFold(n_splits=num_folds)
    accuracies = []
    total_f1 = []
    precisions = []
    recalls = []
    loss = []

    for train_index, eval_index in kf.split(qa_data):
        nth_fold += 1

        train_data = df.iloc[np.r_[train_index]]
        eval_data = df.iloc[np.r_[eval_index]]

        # Initialize the model
        model = ClassificationModel(
            model_type,
            pretrained_model,
            num_labels=2,
            args=model_args,
            use_cuda=True
        )

        # Train the model
        model.train_model(train_data, verbose=False)

        # Evaluate the model
        result, model_outputs, wrong_predictwions = model.eval_model(eval_data, acc=sklearn.metrics.accuracy_score, f1=f1_score_none, precision=precision_score_none, recall=recall_score_none)

        logging.info('{} fold'.format(nth_fold))
        logging.info(result)

        accuracies.append(result['acc'])
        total_f1.append(result['f1'])
        precisions.append(result['precision'])
        recalls.append(result['recall'])
        loss.append(result['eval_loss'])

    logging.info('acc: {} +- {}'.format(np.average(accuracies), np.std(accuracies)))
    logging.info('f1: {} +- {}'.format(np.average(total_f1), np.std(total_f1)))
    logging.info('precision: {} +- {}'.format(np.average(precisions),  np.std(precisions)))
    logging.info('recall: {} +- {}'.format(np.average(recalls),  np.std(recalls)))
    logging.info('loss: {} +- {}'.format(np.average(loss), np.std(loss)))
    end = time.time()
    logging.info('exec time: {}'.format(end - start))


df = pd.read_csv('./data/single-sentence-128.csv')


# ----------------------------------------
model_args = ClassificationArgs()
model_args.evaluate_during_training_steps = 1000
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 10
model_args.do_lower_case = True
model_args.no_cache = True
model_args.no_save = True
model_args.save_model_every_epoch = False
model_args.save_eval_checkpoints = False
model_args.train_batch_size = 3
model_args.eval_batch_size = 3
model_args.learning_rate = 1e-5

logging.info('pair-sentence bert-base-uncased')
questionAndAnswer('bert', 'bert-base-uncased', model_args, df)


# ----------------------------------------
model_args = ClassificationArgs()
model_args.evaluate_during_training_steps = 1000
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 10
model_args.do_lower_case = True
model_args.no_cache = True
model_args.no_save = True
model_args.save_model_every_epoch = False
model_args.save_eval_checkpoints = False
model_args.train_batch_size = 3
model_args.eval_batch_size = 3
model_args.learning_rate = 1e-5

logging.info('pair-sentence bert-large-uncased')
questionAndAnswer('bert', 'bert-large-uncased', model_args, df)


# ----------------------------------------
model_args = ClassificationArgs()
model_args.evaluate_during_training_steps = 1000
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 10
model_args.do_lower_case = True
model_args.no_cache = True
model_args.no_save = True
model_args.save_model_every_epoch = False
model_args.save_eval_checkpoints = False
model_args.train_batch_size = 3
model_args.eval_batch_size = 3
model_args.learning_rate = 1e-5

logging.info('pair-sentence ct-bert')
questionAndAnswer('bert', 'digitalepidemiologylab/covid-twitter-bert-v2', model_args, df)


# ----------------------------------------
model_args = ClassificationArgs()
model_args.evaluate_during_training_steps = 1000
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 10
model_args.do_lower_case = True
model_args.no_cache = True
model_args.no_save = True
model_args.save_model_every_epoch = False
model_args.save_eval_checkpoints = False
model_args.train_batch_size = 3
model_args.eval_batch_size = 3
model_args.learning_rate = 1e-5

logging.info('pair-sentence distilbert')
questionAndAnswer('distilbert', 'distilbert-base-uncased', model_args, df)


# ----------------------------------------
model_args = ClassificationArgs()
model_args.evaluate_during_training_steps = 1000
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 10
model_args.do_lower_case = True
model_args.no_cache = True
model_args.no_save = True
model_args.save_model_every_epoch = False
model_args.save_eval_checkpoints = False
model_args.train_batch_size = 3
model_args.eval_batch_size = 3
model_args.learning_rate = 1e-5

logging.info('pair-sentence roberta-base')
questionAndAnswer('roberta', 'roberta-base', model_args, df)
