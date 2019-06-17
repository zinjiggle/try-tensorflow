

import tensorflow as tf

tf.reset_default_graph()

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')

y = tf.multiply(w, x, name='output')

y_ = tf.constant(0.0)

loss = (y - y_)**2

optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)
grads_and_vars = optim.compute_gradients(loss)

sess = tf.Session()

writer = tf.summary.FileWriter('./', sess.graph)

sess.run(tf.initialize_all_variables())
sess.run(grads_and_vars[0][0])

