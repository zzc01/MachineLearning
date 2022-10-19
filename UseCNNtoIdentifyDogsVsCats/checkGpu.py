#
# Check if GPU is used 
# ref1: https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session
# ref2: https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell
#

if 0: 
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
    # sess = tf.compat.v1.Session()
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

if 1:
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

    with tf.compat.v1.Session() as sess:
        print("Run sess c")
        print (sess.run(c))

if 0:
    tf.debugging.set_log_device_placement(True)
    try:
        with tf.device('/gpu:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
    except RuntimeError as e:
        print(e)

if 0: 
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        inputs = tf.keras.layers.Input(shape=(1,))
        predictions = tf.keras.layers.Dense(1)(inputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
        model.compile(loss='mse',
                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))
