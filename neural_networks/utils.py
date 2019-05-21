import tensorflow as tf
import numpy as np

EPS = 1e-12

def freeze_session(session, keep_var_names=None, output_names=None, quantize=True):
    # https://github.com/amir-abdi/keras_to_tensorflow/blob/master/keras_to_tensorflow.py
    from tensorflow.tools.graph_transforms import TransformGraph
    from tensorflow.python.framework import graph_util
    transforms = []
    if quantize:
        transforms = ["quantize_weights", "quantize_nodes"]
    transformed_graph_def = TransformGraph(
        session.graph.as_graph_def(), [], output_names, transforms
    )
    constant_graph = graph_util.convert_variables_to_constants(
            session,
            transformed_graph_def,
            output_names
    )
    return constant_graph


def softmax2classes(pred, height=256, width=256, num_classes=21):
    pred_mask = np.zeros_like(pred)
    for _i in range(height):
        for _j in range(width):
            cur_top = -1
            cur_max = -1
            for _cl in range(num_classes):
                if pred[_i, _j, _cl] > cur_max:
                    cur_max = pred[_i, _j, _cl]
                    cur_top = _cl
            pred_mask[_i, _j, cur_top] = 1.
    return pred_mask


def get_iou( gt , pr , n_classes=21):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum(( gt[:, :, cl] == 1. )*( pr[:, :, cl] == 1. ))
        union = np.sum(np.maximum( ( gt[:, :, cl] == 1. ) , ( pr[:, :, cl] == 1. ) ))
        iou = float(intersection + EPS)/( union + EPS )
        class_wise[ cl ] = iou
    IoU = np.mean([i for i in class_wise[1:] if i != 1.])
    return IoU, class_wise


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return graph
