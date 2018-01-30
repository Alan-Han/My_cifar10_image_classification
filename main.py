#!/usr/bin/python3
import pickle
import random
import tensorflow as tf

from My_cifar10_image_classification import helper, input_dataset
from My_cifar10_image_classification.cnn import conv_net


def train():
    """Train CIFAR-10 for a number of steps."""
    # Remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # Inputs
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Model
    logits = conv_net(x, keep_prob)

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # Hyperparameters
    epochs = 10
    batch_size = 128
    keep_probability = 0.5

    save_model_path = './results/image_classification'
    valid_features, valid_labels = pickle.load(open('preprocess_batch/preprocess_validation.p', mode='rb'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in helper.get_batches(batch_i, batch_size):
                    sess.run(optimizer, feed_dict={x: batch_features, y: batch_labels, keep_prob: keep_probability})
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                cost_, accuracy_ = sess.run([cost, accuracy],
                                            feed_dict={x: valid_features, y: valid_labels, keep_prob: 1})
                print("cost: {:.3f}, accuracy: {:.3f}".format(cost_, accuracy_))
        saver = tf.train.Saver()
        saver.save(sess, save_model_path)


def test_model():
    """
    Test the saved model against the test dataset
    """

    save_model_path = './results/image_classification'
    test_features, test_labels = pickle.load(open('preprocess_batch/preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    n_samples = 4
    top_n_predictions = 3

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        test_batch_acc_total = sess.run(
            loaded_acc,
            feed_dict={loaded_x: test_features, loaded_y: test_labels, loaded_keep_prob: 1.0})

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


def main():

    input_dataset.cifar10_input()

    train()

    test_model()


if __name__ == '__main__':
    main()