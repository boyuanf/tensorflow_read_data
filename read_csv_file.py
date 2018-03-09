import tensorflow as tf

filename_queue = tf.train.string_input_producer(["C:\\Boyuan\\MyPython\\MNIST_Dataset\\file0.csv",
                                                 "C:\\Boyuan\\MyPython\\MNIST_Dataset\\file1.csv"],
                                                num_epochs=2)

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(12):
        # Retrieve a single instance:
        example, label = sess.run([features, col5])
        print(example, label)


    coord.request_stop()
    coord.join(threads)