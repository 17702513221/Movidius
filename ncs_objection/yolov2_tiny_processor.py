#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# Object detector using SSD Mobile Net

from mvnc import mvncapi as mvnc
import numpy as numpy
import cv2
import time
import threading


class Yolov2_tiny_Processor:

    # Neural network assumes input images are these dimensions.
    YOLO_NETWORK_IMAGE_WIDTH = 416
    YOLO_NETWORK_IMAGE_HEIGHT = 416

    def __init__(self, network_graph_filename: str, ncs_device: mvnc.Device,
                 inital_box_prob_thresh: float, classification_mask:list=None,
                 name = None):
        """Initializes an instance of the class

        :param network_graph_filename: is the path and filename to the graph
               file that was created by the ncsdk compiler
        :param ncs_device: is an open ncs device object to use for inferences for this graph file
        :param inital_box_prob_thresh: the initial box probablity threshold. between 0.0 and 1.0
        :param classification_mask: a list of 0 or 1 values, one for each classification label in the
        _classification_mask list.  if the value is 0 then the corresponding classification won't be reported.
        :param name: A name to use for the processor.  Nice to have when debugging multiple instances
        on multiple threads
        :return : None
        """
        self._device = ncs_device
        self._network_graph_filename = network_graph_filename
        # Load graph from disk and allocate graph.
        try:
            with open(self._network_graph_filename, mode='rb') as graph_file:
                graph_in_memory = graph_file.read()
            self._graph = mvnc.Graph("Yolov2-tiny Graph")
            self._fifo_in, self._fifo_out = self._graph.allocate_with_fifos(self._device, graph_in_memory)

            self._input_fifo_capacity = self._fifo_in.get_option(mvnc.FifoOption.RO_CAPACITY)
            self._output_fifo_capacity = self._fifo_out.get_option(mvnc.FifoOption.RO_CAPACITY)

        except:
            print('\n\n')
            print('Error - could not load neural network graph file: ' + network_graph_filename)
            print('\n\n')
            raise

        self._classification_labels=Yolov2_tiny_Processor.get_classification_labels()

        self._box_probability_threshold = inital_box_prob_thresh
        self._classification_mask=classification_mask
        if (self._classification_mask is None):
            # if no mask passed then create one to accept all classifications
            self._classification_mask = [1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1]

        self._end_flag = True
        self._name = name
        if (self._name is None):
            self._name = "no name"

        # lock to let us count calls to asynchronus inferences and results
        self._async_count_lock = threading.Lock()
        self._async_inference_count = 0

    def cleanup(self, destroy_device=False):
        """Call once when done with the instance of the class

        :param destroy_device: pass True to close and destroy the neural compute device or
        False to leave it open
        :return: None
        """

        self._drain_queues()
        self._fifo_in.destroy()
        self._fifo_out.destroy()
        self._graph.destroy()

        if (destroy_device):
            self._device.close()
            self._device.destroy()


    def get_device(self):
        '''Get the device this processor is using.

        :return:
        '''
        return self._device

    def get_name(self):
        '''Get the name of this processor.

        :return:
        '''
        return self._name

    def drain_queues(self):
        """ Drain the input and output FIFOs for the processor.  This should only be called
        when its known that no calls to start_async_inference will be made during this method's
        exectuion.

        :return: None
        """
        self._drain_queues()

    @staticmethod
    def get_classification_labels():
        """get a list of the classifications that are supported by this neural network

        :return: the list of the classification strings
        """
        ret_labels = list(['background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'dining table', 'dog', 'horse',
          'motorbike', 'person', 'potted plant',
          'sheep', 'sofa', 'train', 'tvmonitor'])
        return ret_labels


    def start_aysnc_inference(self, input_image:numpy.ndarray):
        """Start an asynchronous inference.  When its complete it will go to the output FIFO queue which
           can be read using the get_async_inference_result() method
           If there is no room on the input queue this function will block indefinitely until there is room,
           when there is room, it will queue the inference and return immediately

        :param input_image: he image on which to run the inference.
             it can be any size but is assumed to be opencv standard format of BGRBGRBGR...
        :return: None
        """
        sendtime=time.time()
        # resize image to network width and height
        # then convert to float32, normalize (divide by 255),
        # and finally convert to float16 to pass to LoadTensor as input
        # for an inference
        # this returns a new image so the input_image is unchanged
        input_image = numpy.divide(input_image, 255.0)
        inference_image = cv2.resize(input_image,
                                 (Yolov2_tiny_Processor.YOLO_NETWORK_IMAGE_WIDTH,
                                  Yolov2_tiny_Processor.YOLO_NETWORK_IMAGE_HEIGHT),
                                 cv2.INTER_LINEAR)
        # transpose the image to rgb
        inference_image = inference_image[:, :, ::-1]
        inference_image = inference_image.astype(numpy.float32)
        self._inc_async_count()

        # Load tensor and get result.  This executes the inference on the NCS
        self._graph.queue_inference_with_fifo_elem(self._fifo_in, self._fifo_out, inference_image.astype(numpy.float32), input_image)
        print("send time:",time.time()-sendtime)
        return

    def _inc_async_count(self):
        self._async_count_lock.acquire()
        self._async_inference_count += 1
        self._async_count_lock.release()

    def _dec_async_count(self):
        self._async_count_lock.acquire()
        self._async_inference_count -= 1
        self._async_count_lock.release()

    def _get_async_count(self):
        self._async_count_lock.acquire()
        ret_val = self._async_inference_count
        self._async_count_lock.release()
        return ret_val


    def get_async_inference_result(self):
        """Reads the next available object from the output FIFO queue.  If there is nothing on the output FIFO,
        this fuction will block indefinitiley until there is.

        :return: tuple of the filtered results along with the original input image
        the filtered results is a list of lists. each of the inner lists represent one found object and contain
        the following 6 values:
           string that is network classification ie 'cat', or 'chair' etc
           float value for box X pixel location of upper left within source image
          float value for box Y pixel location of upper left within source image
          float value for box X pixel location of lower right within source image
          float value for box Y pixel location of lower right within source image
          float value that is the probability for the network classification 0.0 - 1.0 inclusive.
        """

        self._dec_async_count()

        # get the result from the queue
        output, input_image = self._fifo_out.read_elem()

        # save original width and height
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]

        # filter out all the objects/boxes that don't meet thresholds
        return self._filter_objects(output, input_image), input_image


    def is_input_queue_empty(self):
        """Determines if the input queue for this instance is empty

        :return: True if input queue is empty or False if not.
        """
        count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        return (count == 0)


    def is_input_queue_full(self):
        """Determines if the input queue is full

        :return: True if the input queue is full and calls to start_async_inference would block
        or False if the queue is not full and start_async_inference would not block
        """
        count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        return ((self._input_fifo_capacity - count) == 0)


    def _drain_queues(self):
        """ Drain the input and output FIFOs for the processor.  This should only be called
        when its known that no calls to start_async_inference will be made during this method's
        exectuion.

        :return: None.
        """
        in_count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
        count = 0

        while (self._get_async_count() != 0):
            count += 1
            if (out_count > 0):
                self.get_async_inference_result()
                out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
            else:
                time.sleep(0.1)

            in_count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
            out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
            if (count > 3):
                blank_image = numpy.zeros((self.YOLO_NETWORK_IMAGE_HEIGHT, self.YOLO_NETWORK_IMAGE_WIDTH, 3),
                                          numpy.float32)
                self.do_sync_inference(blank_image)

            if (count == 30):
                # should really not be nearly this high of a number but working around an issue in the
                # ncapi where an inferece can get stuck in process
                raise Exception("Could not drain FIFO queues for '" + self._name + "'")

        in_count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
        return


    def do_sync_inference(self, input_image:numpy.ndarray):
        """Do a single inference synchronously.
        Don't mix this with calls to get_async_inference_result, Use one or the other.  It is assumed
        that the input queue is empty when this is called which will be the case if this isn't mixed
        with calls to get_async_inference_result.

        :param input_image: the image on which to run the inference it can be any size.
        :return: filtered results which is a list of lists. Each of the inner lists represent one
        found object and contain the following 6 values:
            string that is network classification ie 'cat', or 'chair' etc
            float value for box X pixel location of upper left within source image
            float value for box Y pixel location of upper left within source image
            float value for box X pixel location of lower right within source image
            float value for box Y pixel location of lower right within source image
            float value that is the probability for the network classification 0.0 - 1.0 inclusive.
        """
        self.start_aysnc_inference(input_image)
        filtered_objects, original_image = self.get_async_inference_result()

        return filtered_objects


    def get_box_probability_threshold(self):
        """Determine the current box probabilty threshold for this instance.  It will be between 0.0 and 1.0.
        A higher number means less boxes will be returned.

        :return: the box probability threshold currently in place for this instance.
        """
        return self._box_probability_threshold


    def set_box_probability_threshold(self, value):
        """set the box probability threshold.

        :param value: the new box probability threshold value, it must be between 0.0 and 1.0.
        lower values will allow less certain boxes in the inferences
        which will result in more boxes per image.  Higher values will
        filter out less certain boxes and result in fewer boxes per
        inference.
        :return: None
        """
        self._box_probability_threshold = value

    def sigmoid(self,x):
        return 1.0 / (1 + numpy.exp(x * -1.0))
    def calculate_overlap(self,x1, w1, x2, w2):
        box1_coordinate = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
        box2_coordinate = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
        overlap = box2_coordinate - box1_coordinate
        return overlap
    
    def calculate_iou(self,box, truth):
        # 通过计算重叠高度和宽度计算IOU交会
        width_overlap = self.calculate_overlap(box[0], box[2], truth[0], truth[2])
        height_overlap = self.calculate_overlap(box[1], box[3], truth[1], truth[3])
        # 无重叠
        if width_overlap < 0 or height_overlap < 0:
            return 0

        intersection_area = width_overlap * height_overlap
        union_area = box[2] * box[3] + truth[2] * truth[3] - intersection_area
        iou = intersection_area / union_area
        return iou
    
    def apply_nms(self,boxes):
        # 按降序排序框
        sorted_boxes = sorted(boxes, key=lambda d: d[7])[::-1]
        high_iou_objs = dict()
        # 比较每个检测对象的IOU
        for current_object in range(len(sorted_boxes)):
            if current_object in high_iou_objs:
                continue

            truth = sorted_boxes[current_object]
            for next_object in range(current_object + 1, len(sorted_boxes)):
                if next_object in high_iou_objs:
                    continue
                box = sorted_boxes[next_object]
                iou = self.calculate_iou(box, truth)
                if iou >= 0.30:#IOU_THRESHOLD=0.3
                    high_iou_objs[next_object] = 1

        # 筛选并排序检测项目
        filtered_result = list()
        for current_object in range(len(sorted_boxes)):
            if current_object not in high_iou_objs:
                filtered_result.append(sorted_boxes[current_object])
        return filtered_result


    def _filter_objects(self, output, original_img):
        
        num_classes = 20
        num_grids = 13
        num_anchor_boxes = 5
        original_results = output.astype(numpy.float32)   

        # Tiny Yolo V2 uses a 13 x 13 grid with 5 anchor boxes for each grid cell.
        # This specific model was trained with the VOC Pascal data set and is comprised of 20 classes

        original_results = numpy.reshape(original_results, (13, 13, 125))

        # The 125 results need to be re-organized into 5 chunks of 25 values
        # 20 classes + 1 score + 4 coordinates = 25 values
        # 25 values for each of the 5 anchor bounding boxes = 125 values
        reordered_results = numpy.zeros((13 * 13, 5, 25))
        index = 0
        for row in range( num_grids ):
            for col in range( num_grids ):
                for b_box_voltron in range(125):
                    b_box = row * num_grids + col
                    b_box_num = int(b_box_voltron / 25)
                    b_box_info = b_box_voltron % 25
                    reordered_results[b_box][b_box_num][b_box_info] = original_results[row][col][b_box_voltron]

        # shapes for the 5 Tiny Yolo v2 bounding boxes
        anchor_boxes = [1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52]
        boxes = list()
        classes_boxes_and_probs = []
        # iterate through the grids and anchor boxes and filter out all scores which do not exceed the DETECTION_THRESHOLD
        for row in range(num_grids):
            for col in range(num_grids):
                for anchor_box_num in range(num_anchor_boxes):
                    if self.sigmoid(reordered_results[row * 13 + col][anchor_box_num][4])>self._box_probability_threshold:                        
                        box = list()
                        class_list = list()
                        current_score_total = 0
                        # calculate the coordinates for the current anchor box
                        box_x = (col + self.sigmoid(reordered_results[row * 13 + col][anchor_box_num][0])) / 13.0
                        box_y = (row + self.sigmoid(reordered_results[row * 13 + col][anchor_box_num][1])) / 13.0
                        box_w = (numpy.exp(reordered_results[row * 13 + col][anchor_box_num][2]) *
                                 anchor_boxes[2 * anchor_box_num]) / 13.0
                        box_h = (numpy.exp(reordered_results[row * 13 + col][anchor_box_num][3]) *
                                 anchor_boxes[2 * anchor_box_num + 1]) / 13.0
                
                        # find the class with the highest score
                        for class_enum in range(num_classes):
                            class_list.append(reordered_results[row * 13 + col][anchor_box_num][5 + class_enum])

                        # perform a Softmax on the classes
                        highest_class_score = max(class_list)
                        for current_class in range(len(class_list)):
                            class_list[current_class] = numpy.exp(class_list[current_class] - highest_class_score)

                        current_score_total = sum(class_list)
                        for current_class in range(len(class_list)):
                            class_list[current_class] = class_list[current_class] * 1.0 / current_score_total

                        # probability that the current anchor box contains an item
                        object_confidence = self.sigmoid(reordered_results[row * 13 + col][anchor_box_num][4])
                        # highest class score detected for the object in the current anchor box
                        highest_class_score = max(class_list)
                        # index of the class with the highest score
                        #class_w_highest_score = class_list.index(max(class_list)) + 1
                        class_w_highest_score = self._classification_labels[class_list.index(max(class_list))+1]
                        # the final score for the detected object
                        final_object_score = object_confidence * highest_class_score

                        box.append(box_x)
                        box.append(box_y)
                        box.append(box_w)
                        box.append(box_h)
                        box.append(class_w_highest_score)
                        box.append(object_confidence)
                        box.append(highest_class_score)
                        box.append(final_object_score)

                        # filter out all detected objects with a score less than the threshold
                        if final_object_score > self._box_probability_threshold:
                            boxes.append(box)                        
     
        # gets rid of all duplicate boxes using non-maximal suppression
        results = self.apply_nms(boxes)
        print(results)
        return results
        





