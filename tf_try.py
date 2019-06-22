import tensorflow as tf
import numpy as np

initial_learning_rate = 1e-4

global_step = tf.Variable(0, trainable=False)
num_total_steps = 5*10
boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
values = [initial_learning_rate, initial_learning_rate/2, initial_learning_rate/4]

# learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=10,decay_rate=0.9)
opt = tf.train.AdamOptimizer(learning_rate)

add_global = global_step.assign_add(1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))
    for i in range(50):
        _, rate = sess.run([add_global, learning_rate])
        print(rate)
        print(sess.run([opt._lr]))