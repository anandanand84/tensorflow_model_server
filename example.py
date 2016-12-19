import numpy as np
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('export', False,
                     'If False predicts the pattern from the given imagedata, else train from the existing data'); 

classes          = np.array(['A','B','C','D','E','F','G','H'])
labelscount      = len(classes)
imagesize        = 400;
availableClasses = tf.constant(classes, shape=[labelscount,1], dtype=tf.string);

x = tf.placeholder(tf.float32, [None, imagesize], name="x_placeholder")
W = tf.Variable(tf.zeros([imagesize, labelscount]))
b = tf.Variable(tf.zeros([labelscount]))
pred = tf.nn.softmax(tf.matmul(tf.reshape(x, [-1,400]), W) + b)

init = tf.initialize_all_variables()

saver = tf.train.Saver();

sess = tf.Session();

sess.run(init)

actual = tf.cast(tf.argmax(pred, 1), tf.int32); 
output = tf.reshape(tf.gather(availableClasses, actual), [-1,1])
probability = tf.reshape(tf.reduce_max(tf.nn.softmax(pred), 1), [-1,1])

def export_model(sess, export_path, export_version, x, y , probability):
    export_saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(export_saver)
    model_exporter.init(sess.graph.as_graph_def(), named_graph_signatures={
        'inputs': exporter.generic_signature({'images': x}),
        'outputs': exporter.generic_signature({'classes': y , 'probability' : probability})})
    model_exporter.export(export_path, tf.constant(export_version), sess)
    print("Export Finished!")

def test():
    batch_xs = np.zeros([10,400])
    result, prob =  sess.run([output, probability], feed_dict={x: batch_xs} );
    print('batch xs shape', batch_xs.shape)
    print('Result shape:', result.shape)
    print('probability shape:', prob.shape)
    print('result ', result);
    print('prob ', prob);
if FLAGS.export:    
    export_model(sess, '/model/', 1 , x , output, probability)
else:
    test()
