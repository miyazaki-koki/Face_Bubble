#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import math
import statistics
import datetime

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        start_time = time.perf_counter()

        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.perf_counter() - start_time
        print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)

tDetector =TensoflowFaceDector(PATH_TO_CKPT)

class Detector:
    def __init__(self):
        self.image_path = "test/test.jpg"
        self.before = None
        self.color = (255, 255, 255)
        self.current_point = []
        self.target_point = []
        self.v = []
        self._mask=[]
        self.pix_list = []
        self.dir_list = []
    def face_Detection(self,Rect,h,w):
        image_mask = np.zeros((h,w),np.uint8)
        return  self.Bubble_mask(image_mask,Rect)

    def Bubble_mask(self,_mask,rect):
        R = 50
        _s = math.pow(10,-10)
        for x0,x1,y0,y1 in self._mask:
            _mask[x0:x1,y0:y1] = 255
        height,width = _mask.shape[:2]
        tmp_1 = np.zeros((height,width),np.float32)
        tmp_2 = np.zeros((height,width),np.float32)
        tmp_3 = np.zeros((height,width),np.float32)
        _seg = [np.sqrt(rect[0]**2 + rect[1]**2),rect[1],np.sqrt(rect[1]**2+(width-(rect[0]+rect[2]))**2),
                rect[0],width - (rect[0] + rect[2]),np.sqrt(rect[0]**2+rect[1]**2),
                height - (rect[1] + rect[3]),np.sqrt((width-(rect[0]+rect[2]))**2+(height-(rect[1]+rect[3]))**2)]
        seg_max = np.max(_seg)
        for i , l in enumerate([[0,0],[0,height],[width,0],[width,height]]):
            self.pix_list.append(np.sqrt((l[1]-(rect[1] + rect[3]/2))**2+(l[0]-(rect[0] + rect[2]/2))**2))
        #width
        tmp_3[0:rect[1],0:rect[0]] = _seg[0]
        tmp_3[0:rect[1],rect[0]:rect[0]+rect[2]] = _seg[1]
        tmp_3[0:rect[1],rect[0]+rect[2]:width] = _seg[2]
        tmp_3[rect[1]:rect[1]+rect[3],0:rect[0]] = _seg[3]
        tmp_3[rect[1]:rect[1]+rect[3],rect[0]+rect[2]:width] = _seg[4]
        tmp_3[rect[1]+rect[3]:height,0:rect[0]] = _seg[5]
        tmp_3[rect[1]+rect[3]:height,rect[0]:rect[0]+rect[2]] = _seg[6]
        tmp_3[rect[1]+rect[3]:height,rect[0]+rect[2]:width] = _seg[7]
        for h in range(0,height,R):
            for w in range(0,width,R):
                #distance
                tmp_1[h:h+R,w:w+R] = self.tone((np.sqrt((h-(rect[1] + rect[3]/2))**2+(w-(rect[0] + rect[2]/2))**2)/(np.max(self.pix_list))))
                #direction
                u = (w+R/2) - (rect[0] + rect[2]/2)
                v = (rect[1] + rect[3]/2) - (h+R/2)
                u_v = np.rad2deg(np.arctan2(v,(u+_s)))
                tmp_2[h:h+R,w:w+R] = (1 + -np.sin(np.deg2rad(u_v))) /2
                tmp_3[h:h+R,w:w+R] = 1 - (tmp_3[h:h+R,w:w+R] / seg_max)
                #cost function
                _mask[h:h+R,w:w+R] = 255*(0.4 * tmp_1[h:h+R,w:w+R] + 0.35 * tmp_2[h:h+R,w:w+R] + 0.35 * tmp_3[h:h+R,w:w+R])
        _mask[rect[1]:rect[1] + rect[3],rect[0]:rect[0] + rect[2]] = 255
        self._mask.append([rect[1],rect[1] + rect[3],rect[0],rect[0] + rect[2]])
        hoge = self.Speech_Bubble(_mask)
        self._mask.append([hoge[0] * 50 ,hoge[0] * 50 + 160 ,hoge[1] * 50 ,hoge[1] * 50 + 320])
        _mask[hoge[0] * 50 : hoge[0] * 50 + 160 ,hoge[1] * 50 : hoge[1] * 50 + 320] = 255
        return hoge

    def Speech_Bubble(self,_mask):
        S =50
        kernel_3 = np.ones((160,320),np.uint8)
        height,width = _mask.shape[:2]; fheight,fwidth = kernel_3.shape[:2]
        tmp = np.zeros((int((height-fheight)/S) + 1,int((width-fwidth)/S) + 1),np.float32)
        oh,ow = tmp.shape[:2]
        for h in range(0,oh):
            for w in range(0,ow):
                tmp[h,w]=np.sum(_mask[h*S:h*S+fheight,w*S:w*S+fwidth]*kernel_3)
        _index=np.argwhere(tmp == tmp.min())[0]
        return _index

    def tone(self,x):
        return 0.0001**(math.e**(-3*x))

    def Holo_cam(self, image_path):
        image,_point = self.face_Detection(cv2.imread(image_path))
        cv2.imwrite("./test/detected{}.jpg".format(datetime.time()), image)
        return _point,

def Main(img_path):
    _d = Detector()
    image = cv2.imread(img_path)
    #image = cv2.resize(image,(504,896)) #デバッグ用
    if image is None:
        faces=[]
        Rects=[]
    else:
        [h, w] = image.shape[:2]
        #image = cv2.flip(image, 1)

        (boxes, scores, classes, num_detections) = tDetector.run(image)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)

        P_D=vis_util.position_data
        faces=[]
        Rects=[]
        for i in range(len(P_D.img_L_x)):
            faces.append([P_D.img_L_x[i],P_D.img_L_y[i]
            ,vis_util.position_data.img_width[i],vis_util.position_data.img_height[i],h,w])

            _point = _d.face_Detection([int(P_D.img_L_x[i]),int(P_D.img_L_y[i])
            ,int(vis_util.position_data.img_width[i]),int(vis_util.position_data.img_height[i])],h,w)
            Rects.append([_point[0]*50,_point[1]*50,320,160,h,w])
        P_D.List_reset()
    return faces,Rects
if __name__ == "__main__":
    import collections as cl
    import json

    print("Waiting")
    faces_list,rect_list=Main("./test.jpg")
    data=cl.OrderedDict()

    data["num"]=len(faces_list)
    for i,face in enumerate(faces_list):
        data["x_position"+str(i)]=(int(face[0])-int(int(face[5])/2))
        data["y_position"+str(i)]=(int(face[1])-int(int(face[4])/2))*-1
        data["x_length"+str(i)]=int(face[2])
        data["y_length"+str(i)]=int(face[3])
        data["center_position_x"+str(i)]=((int(face[0])-int(int(face[5])/2)))+int(int(face[2])/2)
        data["center_position_y"+str(i)]=((int(face[1])-int(int(face[4])/2))*-1)-int(int(face[3])/2)

    data["rect_num"]=len(rect_list)
    for i,rect in enumerate(rect_list):
        data["rect_x_position"+str(i)]=((int(rect[0])-int(int(rect[5])/2)))
        data["rect_y_position"+str(i)]=((int(rect[1])-int(int(rect[4])/2))*-1)
        data["rect_x_length"+str(i)]=int(rect[2])
        data["rect_y_length"+str(i)]=int(rect[3])
        data["rect_center_position_x"+str(i)]=((int(rect[0])-int(int(rect[5])/2)))+int(int(rect[2])/2)
        data["rect_center_position_y"+str(i)]=((int(rect[1])-int(int(rect[4])/2))*-1)-int(int(rect[3])/2)
    print("{}".format(json.dumps(data,indent=4)))
