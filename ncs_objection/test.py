#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

from mvnc import mvncapi as mvnc
import cv2
import numpy as np
import sys
import time
import argparse

#Argument parser
arg = argparse.ArgumentParser()
arg.add_argument("-m", "--mode", required=False, type=str, default="image", help="Mode of Neural Network, options: image, video")
arg.add_argument("-n", "--num", required=False, type=int, default=1, help="Number of NCS you want to use")
arg.add_argument("-i", "--image", required=False, type=str, default="test/Traffic.jpg",help="The path to the image you want to process")
arg.add_argument("-v", "--video", required=False, type=str, default="test/bus_station_6094_960x540.mp4",help="The path to the video you want to process or enter a integer if you want to use your webcam")
arg.add_argument("-c", "--score", required=False, type=float, default=0.3, help="compare score")
args = vars( arg.parse_args() )

cv_window_name = "yolov2_tiny"
IMAGE_FROM_DISK = args["image"]
GRAPH_PATH = "model/tiny_yolo_v2.graph"
DETECTION_THRESHOLD = args["score"]         
IOU_THRESHOLD = 0.30

label_name = {0: "bg", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat",
              9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
              16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}


def sigmoid(x):
    return 1.0 / (1 + np.exp(x * -1.0))


def calculate_overlap(x1, w1, x2, w2):
    box1_coordinate = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    box2_coordinate = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    overlap = box2_coordinate - box1_coordinate
    return overlap


def calculate_iou(box, truth):
    # 通过计算重叠高度和宽度计算IOU交会
    width_overlap = calculate_overlap(box[0], box[2], truth[0], truth[2])
    height_overlap = calculate_overlap(box[1], box[3], truth[1], truth[3])
    # 无重叠
    if width_overlap < 0 or height_overlap < 0:
        return 0

    intersection_area = width_overlap * height_overlap
    union_area = box[2] * box[3] + truth[2] * truth[3] - intersection_area
    iou = intersection_area / union_area
    return iou


def apply_nms(boxes):
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
            iou = calculate_iou(box, truth)
            if iou >= IOU_THRESHOLD:
                high_iou_objs[next_object] = 1

    # 筛选并排序检测项目
    filtered_result = list()
    for current_object in range(len(sorted_boxes)):
        if current_object not in high_iou_objs:
            filtered_result.append(sorted_boxes[current_object])
    return filtered_result


def post_processing(output, original_img):

    num_classes = 20
    num_grids = 13
    num_anchor_boxes = 5
    original_results = output.astype(np.float32)   

    # Tiny Yolo V2 uses a 13 x 13 grid with 5 anchor boxes for each grid cell.
    # This specific model was trained with the VOC Pascal data set and is comprised of 20 classes

    original_results = np.reshape(original_results, (13, 13, 125))

    # The 125 results need to be re-organized into 5 chunks of 25 values
    # 20 classes + 1 score + 4 coordinates = 25 values
    # 25 values for each of the 5 anchor bounding boxes = 125 values
    reordered_results = np.zeros((13 * 13, 5, 25))

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
    # iterate through the grids and anchor boxes and filter out all scores which do not exceed the DETECTION_THRESHOLD
    for row in range(num_grids):
        for col in range(num_grids):
            for anchor_box_num in range(num_anchor_boxes):
                box = list()
                class_list = list()
                current_score_total = 0
                # calculate the coordinates for the current anchor box
                box_x = (col + sigmoid(reordered_results[row * 13 + col][anchor_box_num][0])) / 13.0
                box_y = (row + sigmoid(reordered_results[row * 13 + col][anchor_box_num][1])) / 13.0
                box_w = (np.exp(reordered_results[row * 13 + col][anchor_box_num][2]) *
                         anchor_boxes[2 * anchor_box_num]) / 13.0
                box_h = (np.exp(reordered_results[row * 13 + col][anchor_box_num][3]) *
                         anchor_boxes[2 * anchor_box_num + 1]) / 13.0
                
                # find the class with the highest score
                for class_enum in range(num_classes):
                    class_list.append(reordered_results[row * 13 + col][anchor_box_num][5 + class_enum])

                # perform a Softmax on the classes
                highest_class_score = max(class_list)
                for current_class in range(len(class_list)):
                    class_list[current_class] = np.exp(class_list[current_class] - highest_class_score)

                current_score_total = sum(class_list)
                for current_class in range(len(class_list)):
                    class_list[current_class] = class_list[current_class] * 1.0 / current_score_total

                # probability that the current anchor box contains an item
                object_confidence = sigmoid(reordered_results[row * 13 + col][anchor_box_num][4])
                # highest class score detected for the object in the current anchor box
                highest_class_score = max(class_list)
                # index of the class with the highest score
                class_w_highest_score = class_list.index(max(class_list)) + 1
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
                if final_object_score > DETECTION_THRESHOLD:
                    boxes.append(box)

    # gets rid of all duplicate boxes using non-maximal suppression
    results = apply_nms(boxes)

    image_width = original_img.shape[1]
    image_height = original_img.shape[0]

    # calculate the actual box coordinates in relation to the input image
    for box in results:
        box_xmin = (box[0] - box[2] / 2.0) * image_width
        box_xmax = (box[0] + box[2] / 2.0) * image_width
        box_ymin = (box[1] - box[3] / 2.0) * image_height
        box_ymax = (box[1] + box[3] / 2.0) * image_height
        # ensure the box is not drawn out of the window resolution
        if box_xmin < 0:
            box_xmin = 0
        if box_xmax > image_width:
            box_xmax = image_width
        if box_ymin < 0:
            box_ymin = 0
        if box_ymax > image_height:
            box_ymax = image_height

        print (label_name[box[4]], box_xmin, box_ymin, box_xmax, box_ymax)

        # label shape and colorization
        label_text = label_name[box[4]] + " " + str("{0:.2f}".format(box[5]*box[6]))
        label_background_color = (70, 120, 70) # grayish green background for text
        label_text_color = (255, 255, 255)   # white text

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = int(box_xmin)
        label_top = int(box_ymin) - label_size[1]
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]

        # set up the colored rectangle background for text
        cv2.rectangle(original_img, (label_left - 1, label_top - 5),(label_right + 1, label_bottom + 1),
                      label_background_color, -1)
        # set up text
        cv2.putText(original_img, label_text, (int(box_xmin), int(box_ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    label_text_color, 1)
        # set up the rectangle around the object
        cv2.rectangle(original_img, (int(box_xmin), int(box_ymin)), (int(box_xmax), int(box_ymax)), (0, 255, 0), 2)

    return original_img

def compare():
    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)

    # enumerate all devices
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    # use the first device found
    device = mvnc.Device(devices[0])
    # open the device
    device.open()

    # load the model from the disk
    with open(GRAPH_PATH, mode='rb') as f:
        graph_in_memory = f.read()

    graph = mvnc.Graph(GRAPH_PATH)
    # create the input and output fifos
    fifo_in, fifo_out = graph.allocate_with_fifos(device, graph_in_memory)
    
    fps = 0.0
    #Webcam mode, else video file mode
    if args["video"].isdigit():
	    args["video"] = int( args["video"]) 
    cap = cv2.VideoCapture(args["video"])
    if args["mode"] == "image" :
        # read an image in bgr format
        img = cv2.imread(IMAGE_FROM_DISK)
        original_img = img

        # bgr input scaling
        img = np.divide(img, 255.0)
        resized_img = cv2.resize(img, (416, 416), cv2.INTER_LINEAR)

        # transpose the image to rgb
        resized_img = resized_img[:, :, ::-1]
        resized_img = resized_img.astype(np.float32)

        # make an inference
        graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, resized_img, 'user object')
        # get the result
        output, userobj = fifo_out.read_elem()

        # Tiny Yolo V2 requires post processing to filter out duplicate objects and low score objects
        # After post processing, the app will display the image and any detected objects
        post_processing(output, original_img)
        
        cv2.imshow('Tiny Yolo V2', original_img)
        cv2.waitKey()
        
    while args["mode"] == "video":
            start = time.time()
            ret, display_image = cap.read()
            if not ret: 
                    print( "No image found from source, exiting" )
                    break
            original_img = display_image

            # bgr input scaling
            display_image = np.divide(display_image, 255.0)
            resized_img = cv2.resize(display_image, (416, 416), cv2.INTER_LINEAR)

            # transpose the image to rgb
            resized_img = resized_img[:, :, ::-1]
            resized_img = resized_img.astype(np.float32)
            # make an inference
            graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, resized_img, 'user object')
            # get the result
            output, userobj = fifo_out.read_elem()
            # Tiny Yolo V2 requires post processing to filter out duplicate objects and low score objects
            # After post processing, the app will display the image and any detected objects	
            output_image= post_processing(output,original_img)

            fps  = ( fps + ( 1 / (time.time() - start) ) ) / 2
            output_image = cv2.putText( output_image, "fps: {:.1f}".format(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, 4 )

            cv2.imshow( cv_window_name, output_image )
	
            if cv2.getWindowProperty( cv_window_name, cv2.WND_PROP_ASPECT_RATIO ) < 0.0:
                    print( "Window closed" )
                    break
            elif cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
                    print( "Q pressed" )
                    break
    cap.release()
    cv2.destroyAllWindows()
    # clean up
    fifo_in.destroy()
    fifo_out.destroy()
    graph.destroy()
    device.close()
    device.destroy()
    print("Finished")
	   

def main():
    compare()

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
