import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.layers import core as layers_core
import os
import numpy as np
# tf_seq2seq_QA.
from tf_seq2seq_QA.pre_process import DataPATH, indexDataLoad, SimpleDict
from tf_seq2seq_QA.data_set import dataSet, TimeTick


# 准备工作：词向量->转化语料 python pre_process to get tf_seq2seq_QA data.

def createSeq2SeqModel(Xin, Yout):
    pass


if __name__ == "__main__":
    qsdata = indexDataLoad('SQuAD_Question_index_w2v_100D100l')
    paradata = indexDataLoad('SQuAD_Paragraphy_index_w2v_100D100l')

    dictionary = SimpleDict('wiki.en.text.100d100lit5wdVector', embeddingLoad=True)
    embed_wight_init = dictionary.embedding.astype(np.float32)
    vocabSize = dictionary.vocab_size
    decode_input = dictionary.BatchSizeAddStartSign(qsdata)

    learning_rate = 0.0001
    batchsize = 32

    TFDtype = tf.float32
    TFDtypeINOUT = tf.int32
    RNN_SIZE = 256
    RNN_STACK_NUM = 2
    max_gradient_norm = 1

    Numepoch = 10000
    SaveStep = 100
    Numshowloss = 10
    RNNinput_keep_prob = 1
    RNNoutput_keep_prob = 0.7
    RNNstate_keep_prob = 0.7

    data = dataSet((paradata, qsdata, decode_input), batchsize)
    modelpath = os.path.join(os.getcwd(), 'model')
    modelname = os.path.join(modelpath, "seq2seqtest1model.ckpt")
    logpath = os.path.join(os.getcwd(), 'log')

    QuestionSeqMaxLen = qsdata.shape[1]
    ParagraphySeqMaxLen = paradata.shape[1]

    with tf.variable_scope("dynamic_seq2seq", dtype=TFDtype):

        X_paraseq = tf.placeholder(TFDtypeINOUT, shape=[batchsize, ParagraphySeqMaxLen])
        Y_quesseq = tf.placeholder(TFDtypeINOUT, shape=[batchsize, QuestionSeqMaxLen])
        decode_Y_quesseq = tf.placeholder(TFDtypeINOUT, shape=[batchsize, QuestionSeqMaxLen + 1])

        Encode_RNNChoiceLen = tf.placeholder(tf.int32, shape=[batchsize])
        Deccode_RNNChoiceLen = tf.placeholder(tf.int32, shape=[batchsize])

        embed_wight = tf.get_variable('embed_wight', initializer=embed_wight_init, trainable=True, dtype=TFDtype)

        with tf.variable_scope("encoder", dtype=TFDtype) as scope:
            encoder_emb_inp = tf.nn.embedding_lookup(embed_wight, X_paraseq, name='encoder_emb_inp')

            encoder_rnn_cell_list = [tf.nn.rnn_cell.LSTMCell(RNN_SIZE) for i in range(RNN_STACK_NUM)]
            encoder_rnn_cell_list = list(map(
                lambda x: tf.nn.rnn_cell.DropoutWrapper(x, input_keep_prob=RNNinput_keep_prob,
                                                        output_keep_prob=RNNoutput_keep_prob,
                                                        state_keep_prob=RNNstate_keep_prob,
                                                        variational_recurrent=False,
                                                        dtype=TFDtype
                                                        ), encoder_rnn_cell_list))
            encoder_rnn_cell_mul = tf.nn.rnn_cell.MultiRNNCell(encoder_rnn_cell_list)
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_rnn_cell_mul, encoder_emb_inp,
                                                                Encode_RNNChoiceLen,
                                                                time_major=False, swap_memory=True, dtype=TFDtype)
            # encoder_outputs, encoder_states=bidirectional_dynamic_rnn(encoder_rnn_cell, encoder_emb_inp, Encode_RNNChoiceLen,
            #                                                      time_major=False, swap_memory=True)

        projection_layer = layers_core.Dense(vocabSize, use_bias=False, activation=tf.nn.relu)
        with tf.variable_scope("decoder") as scope:
            decoder_emb_inp = tf.nn.embedding_lookup(embed_wight, decode_Y_quesseq, name='decoder_emb_inp')
            decoder_rnn_cell_list = [tf.nn.rnn_cell.LSTMCell(RNN_SIZE) for i in range(RNN_STACK_NUM)]
            decoder_rnn_cell_list = list(map(
                lambda x: tf.nn.rnn_cell.DropoutWrapper(x, input_keep_prob=RNNinput_keep_prob,
                                                        output_keep_prob=RNNoutput_keep_prob,
                                                        state_keep_prob=RNNstate_keep_prob,
                                                        variational_recurrent=False,
                                                        dtype=TFDtype
                                                        ), decoder_rnn_cell_list))
            decoder_rnn_cell_mul = tf.nn.rnn_cell.MultiRNNCell(decoder_rnn_cell_list )
            train_helper = seq2seq.TrainingHelper(decoder_emb_inp, Deccode_RNNChoiceLen, time_major=False,
                                                  name='train_helper')
            decoder = seq2seq.BasicDecoder(decoder_rnn_cell_mul, train_helper, encoder_states,
                                           output_layer=projection_layer)
            decode_outputs, final_context_state, _ = seq2seq.dynamic_decode(decoder, swap_memory=True)

        # createSeq2SeqModel(X_paraseq,Y_quesseq)
        with tf.variable_scope("loss") as scope:
            logits = decode_outputs.rnn_output
            sample_id = decode_outputs.sample_id
            # decode_output_slice = tf.slice(decoder_outputs, [0, 0],
            #                                tf.stack([tf.constant(-1, dtype=tf.int32),
            #                                          tf.cast(tf.reduce_max(tf.argmin(decoder_outputs, 1) + 1), tf.int32)],
            #                                         0))

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.slice(Y_quesseq, begin=[0, 0], size=[batchsize, tf.reduce_max(Deccode_RNNChoiceLen)]),
                logits=logits)
            target_weights = tf.to_float(tf.sequence_mask(Deccode_RNNChoiceLen, tf.reduce_max(Deccode_RNNChoiceLen)))
            train_loss = (tf.reduce_sum(crossent * target_weights) /
                          batchsize)
            tf.summary.scalar('train_loss', train_loss)
        # with tf.variable_scope("metric") as scope:
        #     bleu1=

    # Optimization
    global_step = tf.get_variable("global_step", dtype=tf.int32, initializer=0, trainable=False)
    params = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    update_step = optimizer.apply_gradients(
        zip(clipped_gradients, params), global_step=global_step)

    init = tf.global_variables_initializer()
    save = tf.train.Saver(max_to_keep=2)

    for var in params:
        print(var, var.name)
        tf.summary.histogram(var.name, var)
    merged_summary_op = tf.summary.merge_all()

    print("graph build successfully and  ready to train ...")
    print("local vars")
    for x in tf.local_variables():
        print(x.name)
    print("all vars")
    for x in tf.all_variables():
        print(x.name)
    with tf.Session() as sess:
        train_summary = tf.summary.FileWriter(os.path.join(logpath, 'train'), sess.graph)
        valid_summary = tf.summary.FileWriter(os.path.join(logpath, 'test'), sess.graph)
        ckpt = tf.train.get_checkpoint_state(modelpath)
        if ckpt and ckpt.model_checkpoint_path:
            save.restore(sess, ckpt.model_checkpoint_path)
            orign_step = sess.run(global_step)
            print("already load model...")
        else:
            orign_step = 0
            if not os.path.exists(modelpath):
                os.mkdir(modelpath)
                print("create path %s ..." % (modelpath))
            sess.run(init)
            print("initial var all...")
        TIME = TimeTick()
        timelist = []
        for step in range(1, Numepoch + 1):
            train_X, train_Y, train_TAR = data.next_batch_resample_train()
            if train_X is None or train_Y is None or train_TAR is None:
                pass
            else:
                encoder_rnnchoiclen = dictionary.BatchSizeMaxLen2END(train_X)
                decoder_rnnchoiclen = dictionary.BatchSizeMaxLen2END(train_TAR)
                print("size of sequence with paragraph and  questionare {0:d} and {1:d}".format(encoder_rnnchoiclen.max(),decoder_rnnchoiclen.max()))
                sess.run(update_step, feed_dict={X_paraseq: train_X, decode_Y_quesseq: train_TAR, Y_quesseq: train_Y,
                                                 Encode_RNNChoiceLen: encoder_rnnchoiclen,
                                                 Deccode_RNNChoiceLen: decoder_rnnchoiclen})
                if step % SaveStep == 0:
                    save.save(sess, modelname, global_step)
                    print("already save model")
                if step % Numshowloss == 0:
                    valid_X, valid_Y, valid_TAR = data.next_batch_resample_valid()
                    summary, loss = sess.run([merged_summary_op, train_loss],
                                             feed_dict={X_paraseq: train_X, decode_Y_quesseq: train_TAR,
                                                        Y_quesseq: train_Y,
                                                        Encode_RNNChoiceLen: encoder_rnnchoiclen,
                                                        Deccode_RNNChoiceLen: decoder_rnnchoiclen})
                    train_summary.add_summary(summary, step + orign_step)
                    encoder_rnnchoiclen = dictionary.BatchSizeMaxLen2END(valid_X)
                    decoder_rnnchoiclen = dictionary.BatchSizeMaxLen2END(valid_TAR)
                    valid = sess.run(train_loss,
                                     feed_dict={X_paraseq: valid_X, decode_Y_quesseq: valid_TAR, Y_quesseq: valid_Y,
                                                Encode_RNNChoiceLen: encoder_rnnchoiclen,
                                                Deccode_RNNChoiceLen: decoder_rnnchoiclen})
                    valid_summary.add_summary(summary, step + orign_step)
                    print("epoch %d and loss is %f and validloss is %f ...\n" % (step + orign_step, loss, valid))
                    timelist.append(TIME.time)

        print("mean time is : ", np.mean(timelist))
