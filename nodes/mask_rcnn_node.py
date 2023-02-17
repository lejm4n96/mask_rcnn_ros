#!/usr/bin/env python
import threading
import numpy as np
import resource_retriever

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
from sensor_msgs.msg import RegionOfInterest

from mask_rcnn_ros.mrcnn.config import Config
from mask_rcnn_ros.mrcnn import model as modellib
from mask_rcnn_ros.mrcnn import visualize
from mask_rcnn_ros.msg import Result

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from skimage.transform import resize

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "coco"
    NUM_CLASSES = 1 + 80
    DETECTION_MIN_CONFIDENCE = 0
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 384)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRCNNNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        config = InferenceConfig()
        config.display()

        self._visualization = rospy.get_param('~visualization', True)
        self._visualization_scale = rospy.get_param('~visualization_scale', 1.0)


        # Create model object in inference mode.
        self._model = modellib.MaskRCNN(mode="inference", model_dir="",
                                        config=config)
        # Load weights trained on MS-COCO
        model_path = resource_retriever.get_filename(rospy.get_param('~weight_location'), use_protocol=False)

        rospy.loginfo("Loading pretrained model into memory")
        self._model.load_weights(model_path, by_name=True)
        rospy.loginfo("Successfully loaded pretrained model into memory")

        self._class_names = rospy.get_param('~class_names')

        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._publish_rate = rospy.get_param('~publish_rate', 100)

        # Start ROS publishers
        self._result_pub = \
            rospy.Publisher(
                rospy.get_param('~topic_publishing') + "/result",
                Result,
                queue_size=1
        )

        self._vis_pub = \
            rospy.Publisher(
                rospy.get_param('~topic_publishing') + "/visualization",
                Image,
                queue_size=1
        )

        # Start ROS subscriber
        image_sub = rospy.Subscriber(
            '~cameraTopic',
            Image, 
            self._image_callback,
            queue_size=1
        )

        rospy.loginfo("Running Mask-RCNN...  (Listening to camera topic: '{}')".format(image_sub.name))

    def run(self):
        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                try:
                    rate.sleep()
                except rospy.exceptions.ROSTimeMovedBackwardsException:
                    pass
                continue

            if msg is not None:
                np_image = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

                # Run detection
                results = self._model.detect([np_image], verbose=0)
                result = results[0]
                result_msg = self._build_result_msg(msg, result)
                self._result_pub.publish(result_msg)

                # Visualize results
                if self._visualization:
                    vis_image = self._visualize(result, np_image, self._visualization_scale)
                    cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
                    cv2.convertScaleAbs(vis_image, cv_result)
                    image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                    image_msg.header = msg.header
                    self._vis_pub.publish(image_msg)

            try:
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                pass

    def _build_result_msg(self, msg, result):
        result_msg = Result()
        result_msg.header = msg.header
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            box = RegionOfInterest()
            box.x_offset = x1.item()
            box.y_offset = y1.item()
            box.height = (y2 - y1).item()
            box.width = (x2 - x1).item()
            result_msg.boxes.append(box)

            class_id = result['class_ids'][i]
            result_msg.class_ids.append(class_id)

            class_name = self._class_names[class_id]
            result_msg.class_names.append(class_name)

            score = result['scores'][i]
            result_msg.scores.append(score)

            mask = Int64MultiArray()
            mask_msg = np.zeros(result['masks'].shape[:2], np.int64)
            mask_msg[result['masks'][:,:,i]==True] = np.int64(class_id)
            mask_msg_list = mask_msg.tolist()
            mask.data = [item for sublist in mask_msg_list for item in sublist]
            result_msg.masks.append(mask)
        return result_msg

    def _visualize(self, result, image, scale_factor=1.0):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        # scale input image and masks
        result_resized = result
        height, width = image.shape[:2]
        dim = (int(scale_factor * width), int(scale_factor * height))
        image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        masks = result['masks']
        masks_resized = np.zeros(shape=(dim[1], dim[0], masks.shape[-1]), dtype=masks.dtype)
        for i in range(masks.shape[-1]):
            mask = masks[:, :, i]
            masks_resized[:, :, i] = resize(mask, (dim[1], dim[0]), anti_aliasing=False)

        result_resized['masks'] = masks_resized
        result_resized['rois'] = scale_factor * result['rois']

        # Compute dpi
        dpi = mpl.rcParams['figure.dpi']
        default_height = plt.rcParams['figure.figsize'][1]   # default figure height (4.8 inches)
        height, width = image_resized.shape[:2]

        # Adjust dpi to preserve original image size. This also scales fonts appropriately.
        # This formula works out because height is in pixels (dots) and default_height is in inches,
        # giving dpi (dots per inch).
        dpi = height / default_height

        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = Figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])

        canvas = FigureCanvasAgg(fig)
        display_instances(image_resized, result_resized['rois'], result_resized['masks'],
                          result_resized['class_ids'], self._class_names,
                          result_resized['scores'], ax=ax)
        fig.tight_layout()
        canvas.draw()
        output_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        output_image = output_image.reshape((int(h), int(w), 3))
        return output_image

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):

    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                            alpha=0.7, linestyle="dashed",
                                            edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()


def main():
    rospy.init_node('mask_rcnn')

    node = MaskRCNNNode()
    node.run()

if __name__ == '__main__':
    main()
