import tensorflow as tf
import numpy as np

"""
Name: Leo (Jae-yeon) Yoon
Student ID: 1411513
Date: September 20, 2018
"""
def knn(x_train, y_train, x_test):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 5000 x 784 testing images
    return: predicted y_test which is a 5000 vector
    """
    # TODO: Implement knn here
    
    # 1) Load the Data
    # 2) Initialize the value of k
    # 3.1) Calculate the distance between the test and training data
    # 3.2) Sort the calculated distances in ascending order based on distance values
    # 3.3) Get top k rows from the sorted array
    # 3.4) Get the most frequent class of these rows
    # 3.5) Return the predicted class

    # Initialize Variables/Placeholders
    k = 3
    individual_label = tf.placeholder(dtype=tf.int32)

    tf_x_train = tf.placeholder(dtype=tf.float32,shape=[55000,784])
    tf_y_train = tf.placeholder(dtype=tf.float32,shape=[55000])
    tf_x_test = tf.placeholder(dtype=tf.float32,shape=[784])

    
    #     Accuracy: 0.944000 	Time: 90.640141 (k = 2)
    #     Accuracy: 0.962000 	Time: 90.620143 (k = 3) *BEST*
    #     Accuracy: 0.955000 	Time: 90.846975 (k = 4)
    #     Accuracy: 0.959000 	Time: 91.211091 (k = 5)
    #     Accuracy: 0.951000 	Time: 98.543820 (k = 9)
    #     Accuracy: 0.951000 	Time: 98.926903 (k = 12)

    # Euclidean Distance
    # If we multiply the output of distance by -1, the shortest distance
    # will be the greatest value.
    distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_x_train, tf_x_test)), reduction_indices=1)))
  
    # Finds and returns values and indices of the k largest entries of the last dimension
    # 'distance'. This works because we multiplied the distances by -1, thus
    # the shortest distance will be the greatest values.
    values, indices = tf.nn.top_k(distance,k=k, sorted=False)
    
    # tf.gather returns the label based on the given indices
    predicted_labels = tf.gather(tf_y_train, indices)
    
    # y : unique numbers/elements
    # idx : index relative to the count array
    # count: number of each unique variables
    y, idx, count = tf.unique_with_counts(predicted_labels)
    
    # Prediction based on most recurring number
    the_prediction = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]
    
    the_prediction = tf.cast(the_prediction, dtype=tf.int32)


    prediction_array=[]
    initialize = tf.global_variables_initializer()
    
    with tf.Session() as sess:
      
      for i in range(x_test.shape[0]):
        sess.run(initialize)
        # Load individial cases of x_test by spliting it by the i'th element in
        # the array of x_test
        feed_dict = {tf_x_train: x_train,
                     tf_y_train: y_train,
                     tf_x_test: x_test[i,:]
                    }
        # In the end, the sess.run returns individual labels, so we just append
        # them to the array, and return the array.
        prediction_array.append(sess.run(the_prediction,feed_dict=feed_dict))
        
      return prediction_array
        
    raise NotImplementedError
