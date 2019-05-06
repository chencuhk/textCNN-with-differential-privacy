import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import word2vec_helpers
from tensorflow.python.framework import ops
import math
import chocolate as choco
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .3, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "D:/sentiment/cnn-text-classification-denny/data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "D:/sentiment/cnn-text-classification-denny/data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 240, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 130, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("eps",3.76,"defaut is 2")
tf.flags.DEFINE_float("delta",0.00001,"defaut is 2")
FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")
gradient_sigma=0.0
def data_scale(eps):
    #calculating data scale for training data
    #training data scale is sqrt(dF/(eps*sqrt(2))
    #where dF is the max filter size
    #df=float(FLAGS.filter_sizes.split(",")[-1])
    df=2
    print(df)
    scale=math.sqrt(df/(eps*math.sqrt(2)))
    return scale

def gradient_scale(data_size,eps):
    q=float(FLAGS.batch_size)/float(data_size) #prob of noisy batch in total data size
    T=FLAGS.num_epochs
    log_term1=1/FLAGS.delta
    log_term2=T/FLAGS.delta
    log1=math.log10(log_term1)
    log2=math.log10(log_term2)
    sqrt_term=T*log1*log2
    sigma=math.ceil(q*math.sqrt(sqrt_term)/eps)
    return sigma

def create_space():
    space = {"learning_rate" : choco.log(low=-5, high=-2, base=10),
             
             "dropout_keep_prob" : choco.quantized_uniform(low=0.0, high=0.95,step=0.05),
             "num_filters":choco.quantized_uniform(low=50, high=200, step=10),
             "batch_size":choco.quantized_uniform(low=64, high=256, step=16),
             "num_epochs":choco.quantized_uniform(low=100, high=200, step=10),
             "l2_reg_lambda":choco.quantized_uniform(low=0.0, high=10.0, step=0.5),
             "eps":choco.quantized_uniform(low=1.0, high=10.0,step=0.02),
             "dev_sample_percentage":choco.quantized_uniform(low=0.1, high=0.3,step=0.01)
             }

    return space
def preprocess(eps,dev_rate):
    # Data Preparation
    # ==================================================
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Load data
    print("Loading data...")
    x_text, y,pos_len,neg_len = data_helpers.noisy_load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    data_size=len(x_text)
    """
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    """
    # Get embedding vector
    sentences, max_document_length = data_helpers.padding_sentences(x_text, '<PADDING>')
    x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size = FLAGS.embedding_dim, file_to_save = os.path.join(out_dir, 'trained_word2vec.model')))
    #x=tf.cast(x, tf.float32)
    #vectors =word2vec_helpers.embedding_sentences([['first', 'sentence'], ['second', 'sentence']], embedding_size = 4, min_count = 1)
    print(x[0].shape)
    #y =np.reshape(y,(-1,1))
    print("x.shape = {}".format(x.shape))
    print("y.shape = {}".format(y.shape))
    #adding noise according to different classes
    
    data_sigma=data_scale(eps)
    global gradient_sigma
    gradient_sigma=gradient_scale(data_size,eps)
    pos_noise=np.random.normal(0,data_sigma, [pos_len,x.shape[1],x.shape[2]]) 
    neg_noise=np.random.normal(0,data_sigma, [neg_len,x.shape[1],x.shape[2]])
    noise = np.concatenate([pos_noise, neg_noise], 0)
    x=x+noise
    # Save params
    """
    training_params_file = os.path.join(out_dir, 'training_params.pickle')
    params = {'num_labels': FLAGS.num_labels,'max_document_length' : max_document_length}
    data_helpers.saveDict(params, training_params_file)"""
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    print(shuffle_indices)
    #x_shuffle_indices=[[index] for index in shuffle_indices]
    print("the shape of x:{}".format(x.shape[0]))
    print("indices shape:{}".format(shuffle_indices))
    """
    x_shuffled=tf.gather_nd(
    x,
    x_shuffle_indices,
    name=None
)"""
    x_shuffled = x[shuffle_indices]
    #x_shuffled = x[x_shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    
    #dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    dev_sample_index = -1 * int(dev_rate * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("shape of x:{}".format(x_train.shape))
    print("shape of y:{}".format(y_train.shape))
    del x, y, x_shuffled, y_shuffled
    """
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    """
    #return x_train, y_train, vocab_processor, x_dev, y_dev
    return x_train, y_train,x_dev, y_dev

#def train(x_train, y_train, vocab_processor, x_dev, y_dev):
def train(x_train, y_train, x_dev, y_dev,params):
    # Training
    # ==================================================
    cnn_loss=0.0
    global gradient_simga
    lr=params["learning_rate"]
    num_filters=params["num_filters"]
    batch_size=params["batch_size"]
    num_epochs=params["num_epochs"]
    l2_reg_lambda=params["l2_reg_lambda"]
    #eps=params["eps"]
    #data_size=len(x_train)+len(x_dev)
    print("gradient sigma:{}".format(gradient_sigma))
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            
            
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                #vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=num_filters,
                #num_filters=FLAGS.num_filters,
                #l2_reg_lambda=FLAGS.l2_reg_lambda
                l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            #optimizer = tf.train.AdamOptimizer(1e-3)
            optimizer = tf.train.AdamOptimizer(lr)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)

            # adding gassian noise to gradients
            #grads_and_vars =[(g,v) for g, v in grads_and_vars] 
            """var_list = tf.trainable_variables()
            grad_var_list = optimizer.compute_gradients(loss=cnn.loss, var_list=var_list) """
            #get isolated grad values
            gradient_scale=gradient_sigma
            grad_noise=[]
            for grad,var in grads_and_vars:
                if grad is not None:
                    if isinstance(grad, ops.IndexedSlices):
                        grad_values = grad.values
                        print(type(grad_values))
                        grad_values = grad_values+np.random.normal(loc=0,scale=gradient_scale)
                        grad = ops.IndexedSlices(grad_values, grad.indices, grad.dense_shape)
                    else:
                        grad_values = grad
                        grad_values = grad_values+np.random.normal(loc=0,scale=gradient_scale)
                grad_noise.append((grad,var))
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            """
            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            """
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
                cnn_loss =float(loss)
                #print("type of CNN:{},value:{}".format(type(cnn_loss),cnn_loss))
                return cnn_loss
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), batch_size, num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                cnn_loss=train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
    return cnn_loss
def main(argv=None):
    
    #using the chocolate to tune
    space = create_space()
    conn = choco.SQLiteConnection(url="sqlite:///db2.db")
    cv=choco.Repeat(repetitions=3,reduce=np.mean,rep_col="_repetition_id")
    sampler = choco.Bayes(conn, space,crossvalidation=cv)
    #train(x_train, y_train, vocab_processor, x_dev, y_dev)
    token, params = sampler.next()
    print(type(token))
    print(token)
    x_train, y_train, x_dev, y_dev = preprocess(params["eps"],params["dev_sample_percentage"])
    loss = train(x_train, y_train, x_dev, y_dev,params)
    print(loss)
    sampler.update(token, loss)
    results = conn.results_as_dataframe()
    print(results)
    results = pd.melt(results, id_vars=["_loss"], value_name='value', var_name="variable")
    sns.lmplot(x="value", y="_loss", data=results, col="variable", col_wrap=3, sharex=False)
    plt.show()


if __name__ == '__main__':
    tf.app.run()
