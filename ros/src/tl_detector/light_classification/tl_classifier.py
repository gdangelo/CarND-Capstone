from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np

FROZEN_GRAPH_PATH = 'light_classification/models/ssd_sim/frozen_inference_graph.pb'

class TLClassifier(object):
    def __init__(self):
        # Load a (frozen) Tensorflow model into memory
        self.detection_graph, self.all_tensor_names = self.load_graph()

        # Load tensors needed for inference
        self.num_detections = self.load_tensor('num_detections')
        self.boxes = self.load_tensor('detection_boxes')
        self.scores = self.load_tensor('detection_scores')
        self.classes = self.load_tensor('detection_classes')
        self.image_tensor = self.load_tensor('image_tensor')

        # Minimum threshold for taking detection into account
        self.detection_threshold = 0.5

        # Load TF session
        self.sess = tf.Session(graph=self.detection_graph)

    def load_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(FROZEN_GRAPH_PATH, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

        # Also retrieve all tensor names
        ops = graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}

        return graph, all_tensor_names

    def load_tensor(self, name):
        # Retrieve tensor from graph
        tensor_name = name + ':0'
        if tensor_name in self.all_tensor_names:
            return self.detection_graph.get_tensor_by_name(tensor_name)

        return None

    def convert_class_id_to_traffic_light(self, class_id):
        if class_id == 1:
            return TrafficLight.GREEN
        if class_id == 2:
            return TrafficLight.RED
        if class_id == 3:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Run inference
        if self.detection_graph and self.sess and self.num_detections and self.boxes and self.scores and self.classes:
            with self.detection_graph.as_default():
                num_detections, boxes, scores, classes = self.sess.run(
                    [self.num_detections, self.boxes, self.scores, self.classes],
                    feed_dict={self.image_tensor: np.expand_dims(image, 0)})

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                # Return the detection with highest score if above the min threshold
                detected_class = classes[0] if (num_detections > 0 and scores[0] > self.detection_threshold) else -1
                return self.convert_class_id_to_traffic_light(detected_class)

        return TrafficLight.UNKNOWN
