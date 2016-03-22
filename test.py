import tensorflow as tf

with tf.Graph().as_default(), tf.Session() as sess:
    a=tf.Variable(tf.random_normal([8192, 8192], stddev=0.35))
    b=tf.Variable(tf.random_normal([8192, 8192], stddev=0.35))
    c=tf.matmul(a, b)

    saver = tf.train.Saver()  # defaults to saving all variables
    sess.run([tf.initialize_all_variables()])

    import time
    start_time = time.time()
    sess.run(c)
    print time.time()-start_time

    tf.train.write_graph(sess.graph_def, '.' , 'test.pb', as_text=False)
