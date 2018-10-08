from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np

class TLClassifier(object):
    def __init__(self, model_file, min_score_thresh=.5):
        # Load a (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Minimum threshold for taking detection into account
        self.min_score_thresh = min_score_thresh

    def convert_class_id_to_traffic_light(self, class_id):
        if class_id == 1:
            return TrafficLight.GREEN
        if class_id == 2:
            return TrafficLight.YELLOW
        if class_id == 3:
            return TrafficLight.RED
        else:
            return TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Retrieve all tensors names from graph
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}

                # Build a dict with only the tensors needed for detection results
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                # Process tensors for single image inference
                if 'detection_masks' in tensor_dict:
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

                # Retrieve input image tensor for inference
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                # All outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[ 'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                # Return the detection with highest score if above the min threshold
                detected_class = None
                if output_dict['num_detections'] > 0 and output_dict['detection_scores'][0] > self.min_score_thresh:
                    detected_class = output_dict['detection_classes'][0]

                return self.convert_class_id_to_traffic_light(detected_class)
