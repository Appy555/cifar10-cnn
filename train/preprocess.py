import tensorflow as tf

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)

    print("Testing images:", x_test.shape)
    print("Testing labels:", y_test.shape)

    return x_train, y_train, x_test, y_test 

if __name__ == "__main__":
    load_data()