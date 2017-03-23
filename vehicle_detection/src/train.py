from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import pickle

import tensorflow as tf
from tensorflow.contrib import slim


from sklearn.utils import shuffle
from .preprocessing import *

MODEL_PATH = "saved_models/"
CNN_TRAIN_FOLDER = "training_images/cnn"
CHECK_POINT_NAME = "model.ckpt"

IMAGE_SIZE = 64

# number of epochs
NB_EPOCHS = 50
NB_TRAIN_SAMPLES = 3000
# batch size
BATCH_SIZE = 64
# filter size
FILTER_SIZE = 3
#number of conv filters
NB_FILTERS = 16
#input channels
INPUT_CHANNELS = 3
# number of hidden layers
NB_HIDDEN = 64
# Learning Rate
LR = 0.001
NB_CLASSES = 2


def save_model(model_fit, model_name):
    # Persist model for lookup at test time
    _file = open(MODEL_PATH + model_name ,'wb')
    pickle.dump(model_fit, _file)
    _file.close()
    print("Model {} saved..".format(model_name))

def train_svm_model(X,y):
    t=time.time()

    # Scaler
    scaler = StandardScaler()
    scaler.fit(X)
    save_model(scaler,"standard_scaler")
    X = scaler.transform(X)

    # SVM Fit
    svc = LinearSVC(penalty='l2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svc.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Save the model
    save_model(svc,"linear_svm")

#  Models.
def build_cnn(data, one_hot_y, loss_beta = 0.001):
    # Variables.
    # For saving and optimizer.
    global_step = tf.Variable(0, name="global_step")
    tf.add_to_collection("global_step", global_step)
    # conv weights
    layer1_weights = tf.Variable(tf.truncated_normal(
      [FILTER_SIZE, FILTER_SIZE, INPUT_CHANNELS, NB_FILTERS], stddev=0.1), name="layer1_weights")
    layer1_biases = tf.Variable(tf.zeros([NB_FILTERS]), name="layer1_biases")
    layer2_weights = tf.Variable(tf.truncated_normal(
      [FILTER_SIZE, FILTER_SIZE, NB_FILTERS, NB_FILTERS], stddev=0.1), name="layer2_weights")
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[NB_FILTERS]), name="layer2_biases")
    # fully connected layers
    layer3_weights = tf.Variable(tf.truncated_normal(
      [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * NB_FILTERS, NB_HIDDEN], stddev=0.1), name="layer3_weights")
    l2_layer3 = loss_beta * tf.nn.l2_loss(layer3_weights)
    tf.add_to_collection("loss", l2_layer3)

    layer3_biases = tf.Variable(tf.constant(1.0, shape=[NB_HIDDEN]), name="layer3_biases")
    layer4_weights = tf.Variable(tf.truncated_normal(
      [NB_HIDDEN, NB_CLASSES], stddev=0.1), name="layer4_weights")
    #l2_layer4 = loss_beta * tf.nn.l2_loss(layer4_weights)
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[NB_CLASSES]), name="layer4_biases")

    # Add variables to histogram
    # for variable in slim.get_model_variables():
    #   tf.summary.histogram(variable.op.name, variable)

    # 1st Convolution
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    # Relu activation of conv
    hidden = tf.nn.relu(conv + layer1_biases)
    # Max pooling of activation
    hidden_pooled = tf.nn.max_pool(hidden, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')
    #print(hidden_pooled.get_shape())
    # 2nd Convolution
    conv = tf.nn.conv2d(hidden_pooled, layer2_weights, [1, 1, 1, 1], padding='SAME')
    # Relu activation of conv
    hidden = tf.nn.relu(conv + layer2_biases)
    # Max pooling
    hidden_pooled = tf.nn.max_pool(hidden, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')
    #print(hidden_pooled.get_shape())
    # Flatten
    shape = hidden_pooled.get_shape().as_list()
    reshape = tf.reshape(hidden_pooled, [-1, shape[1] * shape[2] * shape[3]])
    # 1st fully connected layer with relu activation
    #print(reshape.get_shape())
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    # 2nd fully connected layer
    _logits = tf.matmul(hidden, layer4_weights) + layer4_biases
    tf.add_to_collection("logits", _logits)
    return _logits

def train_cnn(gen, X_val, y_val):
    global_step = tf.Variable(0, name="global_step")
    x = tf.placeholder(
        tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, INPUT_CHANNELS))

    tf.add_to_collection("input", x)
    #tf.summary.histogram("model/input_raw", x)

    y = tf.placeholder(tf.int32, shape=(None),name='y')
    tf.add_to_collection("labels", y)

    one_hot_y = tf.one_hot(y, NB_CLASSES)

    # Set the graph computations.
    logits  = build_cnn(x, one_hot_y)

    # loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    _regulization = tf.get_collection("loss")[0]
    loss_operation += _regulization
    #tf.summary.scalar('learning_rate', loss_operation)
    #loss_operation += l2_layer3
    # accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection("accuracy", accuracy_operation)

    optimizer = tf.train.AdamOptimizer(learning_rate = LR)
    training_operation = optimizer.minimize(loss_operation)

    #merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter("summary/", graph=tf.get_default_graph())
    saver = tf.train.Saver()

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples


    vl_accuracy = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Training...")
        print()
        for i in range(NB_EPOCHS):
            tr_batch_error_rate = []
            print("Epoch {} ...".format(i+1))
            for offset in range(0, NB_TRAIN_SAMPLES, BATCH_SIZE):
                batch_x, batch_y = next(gen)
                _, l = sess.run([training_operation, loss_operation] , feed_dict={x: batch_x, y: batch_y})
                # Summary
                #step = tf.get_collection("global_step")[0]
                #writer.add_summary(_summary, step.eval(sess))
                # Error
                tr_batch_error_rate.append(l)
            validation_accuracy = evaluate(X_val, y_val)
            vl_accuracy.append(validation_accuracy)
            print("Batch Training Loss = {:.3f}".format(np.mean(tr_batch_error_rate)))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
            if (i % 5 == 0) and (i > 0):
                saver.save(sess, MODEL_PATH + CHECK_POINT_NAME)
        # Save the variables to disk.
        _file_name = MODEL_PATH + CHECK_POINT_NAME
        saver.save(sess, _file_name)
        print("Final model saved in file: %s" % _file_name)

def train():
    X, y = create_feature_space_for_cnn()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.2)
    def _mean_substract(img):
        #return img - np.mean(img)
        return (img / 255.).astype(np.float32)
    X_val = _mean_substract(X_val)
    gen = ImageGenerator(BATCH_SIZE, X, y)
    train_cnn(gen, X_val, y_val)

# tf.reset_default_graph()
# saver = tf.train.import_meta_graph("saved_models/model.ckpt.meta",clear_devices=True)
# gr = tf.get_default_graph()
# print("Number of ops in TF Graph is {}".format(len(gr.get_operations())))
# _X_val, _y_val = X_val[:50], y_val[:50]
# print(Counter(y_val[:50]), print(_X_val.shape))

# with tf.Session() as sess:
#     gr = tf.get_default_graph()
#     print("Number of ops in TF Graph is {}".format(len(gr.get_operations())))
#     latest_checkpoint = tf.train.latest_checkpoint(MODEL_PATH)
#     print("checkpoint : {}".format(latest_checkpoint))
#     saver.restore(sess, latest_checkpoint)
#     logits_ops = tf.get_collection("logits")[0]
#     print(logits_ops)
#     input_tensor = tf.get_collection("input")[0]
#     print(input_tensor)
#     accuracy_tensor = tf.get_collection("accuracy")[0]
#     y_tensor = gr.get_collection("labels")[0]

#     def evaluate(X_data, y_data):
#         num_examples = len(X_data)
#         total_accuracy = 0
#         for offset in range(0, num_examples, BATCH_SIZE):
#             batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
#             accuracy = sess.run(accuracy_tensor, feed_dict={input_tensor: batch_x, y_tensor: batch_y})
#             total_accuracy += (accuracy * len(batch_x))
#         return total_accuracy / num_examples

#     def predict(_X_val):
#         _predictions =  sess.run([logits_ops], feed_dict={input_tensor: _X_val})
#         #print(_predictions)
#         return _predictions[0].argmax(axis=1)

#     #print(evaluate(_X_val, _y_val))
#     print(predict(_X_val))
