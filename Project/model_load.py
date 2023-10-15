# -*- coding: utf-8 -*-
"""
Script creating class for loading the model which detects the objects (in our project we are
interested only in people figure) and coordinates of their boxes
"""

import numpy as np
import tensorflow as tf


class Model:
    """
    Class that contains the model and all its functions
    code for __init__ and predict methods mostly come from the github repository in the following link
    https://github.com/basileroth75/covid-social-distancing-detection/blob/master/src/tf_model_object_detection.py
    """
    def __init__(self, model_path):
        """
        Initialization function
        model_path : path to the model 
        """

        # Declare detection graph
        self.detection_graph = tf.Graph()
        
        # Load the model into the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as file:
                serialized_graph = file.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Create a session from the detection graph
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

    def predict(self, img):
        """
        Get the prediction results on 1 frame
        img : our img vector
        """
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        img_exp = np.expand_dims(img, axis=0)
        # Pass the inputs and outputs to the session to get the results 
        boxes, scores, classes = self.sess.run([self.detection_graph.get_tensor_by_name('detection_boxes:0'),
                                                self.detection_graph.get_tensor_by_name('detection_scores:0'),
                                                self.detection_graph.get_tensor_by_name('detection_classes:0')],
                                               feed_dict={self.detection_graph.get_tensor_by_name('image_tensor:0'):
                                               img_exp})
        return boxes, scores, classes

    def generate_boxes(self, img):
        """
        method generates boxes for the detected objects on the given image/frame
        """
        
        # we save the output from the 'predict' method
        boxes, scores, classes = self.predict(img)
        
        boxes_real = []

        # we denormalize the coordinates of the boxes and filter them to receive
        # only the interesting ones for our project

        for i, points in enumerate(boxes[0]):
            if classes[0][i] != 1:
                continue
            if scores[0][i] < 0.7:
                break
            pts = points.copy()
            pts[0], pts[2] = pts[0]*img.shape[0], pts[2]*img.shape[0]
            pts[1], pts[3] = pts[1]*img.shape[1], pts[3]*img.shape[1]
            
            boxes_real.append(pts)
        
        return boxes_real
