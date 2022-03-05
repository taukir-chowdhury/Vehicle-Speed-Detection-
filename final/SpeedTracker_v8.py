import torch
import os
import cv2
import numpy as np
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from utils.datasets import LoadImages
import timeit
import csv
from sympy import Point, Polygon, Line
from tqdm import tqdm


start_from_checkpoint = False
checkpoint = 0
### Input and Output Path ###
result_path = "../results"
am_pm = 'am'
TIME_START = '9'
input_name = "2 01 R 14022022090000Am"
video_path = f"../vehicle_data/{input_name}.m4v"
if not os.path.exists(os.path.join(result_path,input_name)):
    os.makedirs(os.path.join(result_path,input_name))
    os.makedirs(os.path.join(result_path,input_name,"saved_images"))
    
else:
    start_from_checkpoint = True
    files = os.listdir(os.path.join(result_path,input_name))
    for i in files:
        if i[-4:] == ".csv":
            namelist = i.split("_")
            name = int(namelist[-1][:-4])
            if name>checkpoint:
                checkpoint = name
    
    
    
output_name = input_name + "_output"
output_video = f"{result_path}/{input_name}/{output_name}.avi"

FPS = 12


### Model Loading ###
model = torch.hub.load("yolov5", "custom", 
                        source="local", path="yolov5/yolov5s.pt", force_reload=True)

deep_sort_model = "osnet_x0_25"
config_deepsort = "deep_sort/configs/deep_sort.yaml"

cfg = get_config()
cfg.merge_from_file(config_deepsort)
deepsort = DeepSort(deep_sort_model,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)




Entered_Polygon = {}
Speed = {}
data =[]
log = []
FRAME_COUNT = 0

# preprocessing
x_numpy = 1914
y_numpy = 1080

x_editor = 1920
y_editor = 1080

LENGTH = 16 * .001 #km or 6 meter

x_factor = x_numpy/x_editor
y_factor = y_numpy/y_editor
distance_cof = 8.7/700


def process_coordinates(area):
    area_processed = []
    for co in area:
        area_processed.append((co[0] * x_factor, co[1] * y_factor ))
    
    return area_processed


area = [(571,291), (930,393),(1000, 460), (510,310), (861,633), (770, 714), (82, 470), (198,423)]
dr1 = [(0,620), (440,975)]
distance_line = [(810,516), (825,520), (400, 1060), (375,1045)]


deleting_region1 = process_coordinates(dr1)
area_processed = process_coordinates(area)
distance_line_processed = process_coordinates(distance_line)

DELETING_LINE = Line(deleting_region1[0],deleting_region1[1])
ENTRY_LINE = Polygon(area_processed[0], area_processed[1], area_processed[2], area_processed[3])
EXIT_LINE = Polygon(area_processed[4], area_processed[5], area_processed[6], area_processed[7])


def if_inside_polygon(center, polygonArea):
    # center = (cx, cy)
    # polygon = [(x1,y1), (x2, y2), (x3, y3), (x4, y4)] 
    result = cv2.pointPolygonTest(np.array(polygonArea, np.int32),center, False)
    # 0 -> on the boundary 
    # -1 -> outside
    # 1 -> inside
    if result >= 0:
        return True
    return False


def get_distance(center, polygon):
    dist = cv2.pointPolygonTest(np.array(polygon, np.int32),center, True)
    return dist*distance_cof



def if_intersect(bbox, line):
    poly1 = Polygon(bbox[0], bbox[1], bbox[2], bbox[3])
    r = poly1.intersection(line)
    if len(r) > 0:
        return True
    return False 


def calculate_speed(entry_time,exit_time):
    time_diff = exit_time - entry_time
    #print(exit_time,entry_time)
    speed = (LENGTH/time_diff) * 3600
    #print(speed)
    return round(speed,3)



def adding_to_report(frame,ID,speed,time, xmin, ymin, xmax, ymax):
    
    #cropped = img[start_row:end_row, start_col:end_col]
    cropped_image = frame[ymin:ymax,xmin:xmax]
    
    cx = int((xmin + xmax)/2)
    cy = int((ymin + ymax)/2)
    d = abs(get_distance((cx,cy),distance_line_processed))
    
    name = f"{time*FPS}_{ID}"
    if int(time/60) == 60:
        time = time % 60
    
    clock_time = f"{int(TIME_START)}:{int(time/60)}:{int(time%60)} {am_pm}"
    cv2.imwrite(f"{os.path.join(result_path,input_name,'saved_images')}/{name}.png", cropped_image)
    data.append([name,clock_time,speed,d])

    
def create_report(data,i):
    header = ['name', 'time(s)', 'speed(km/h)', 'distance(m)']
    with open(f"{result_path}/{input_name}/report_{i}.csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)



def create_log(log):
    with open(f"{result_path}/{input_name}/log.txt", "w") as f:
        text = "".join(log)
        f.write(text)


def if_skip_frame():
    if int(FRAME_COUNT / 3600) % 2 != 0:
        return True   
    return False

def process_frame_for_speed(frame, obj_info,font=cv2.FONT_HERSHEY_SIMPLEX, box_width=1, font_size=1, 
            detected_color=(0, 255, 0), font_color=(255, 255, 255), font_thickness = 1):

    
    for obj in obj_info:
        #top, right, bottom, left
        xmin, ymin, xmax, ymax = obj[0], obj[1], obj[2], obj[3]
        w = xmax - xmin
        h = ymax - ymin
        bbox = [(xmin, ymin),  (xmin+w, ymin), (xmax, ymax), (xmin, ymin+h)]
        
        ID = int(obj[-1])
        # if Entered_Polygon.get(ID, -1) == -1:
        #     cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), color = (255,0,0), thickness = 1)
        # else:
        #     cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), color = (0,0,255), thickness = 1)
        
        text = ""
        log.append(f"ID: {ID}\n")
        if Entered_Polygon.get(ID, -1) == -1 and not if_intersect(bbox, ENTRY_LINE):
            continue
        
        elif Entered_Polygon.get(ID, -1) == -1 and if_intersect(bbox, ENTRY_LINE):
            entry_time = FRAME_COUNT/FPS
            Entered_Polygon[ID] = entry_time
            log.append(f"ID: {ID} crossing entry line, entry_time: {entry_time}, Entered_Polygon_dict: {Entered_Polygon}\n\n")
        
        elif Entered_Polygon.get(ID, -1) != -1 and if_intersect(bbox, EXIT_LINE):
        
            if Speed.get(ID,-1) == -1 :
                exit_time = FRAME_COUNT/FPS
                entry_time = Entered_Polygon.get(ID)
                speed = calculate_speed(entry_time,exit_time)
                
                Speed[ID] = speed
                adding_to_report(frame,ID,speed,exit_time,xmin, ymin, xmax, ymax)
                log.append(f"ID: {ID} crossing exit line, entry_time: {entry_time}, exit_time: {exit_time}\nspeed: {speed}\nEntered_Polygon_dict: {Entered_Polygon}, Speed_dict: {Speed}\n\n")
            
            else:
              speed = Speed[ID]
            
            text = str(speed) + "km/h" 
        

        elif Entered_Polygon.get(ID, -1) != -1 and if_intersect(bbox, DELETING_LINE):
            log.append(f"ID: {ID}- Deleting_Line\n")
            time = FRAME_COUNT/FPS
            log.append(f"Before Deleting:\nntered_Polygon_dict: {Entered_Polygon}, Speed_dict: {Speed}\n")
            
            
            del Entered_Polygon[ID]
            if Speed.get(ID, -1) != -1:
                del Speed[ID]
            #adding_to_report(frame,ID,speed,exit_time,xmin, ymin, xmax, ymax)
            log.append(f"After Deleting:\nntered_Polygon_dict: {Entered_Polygon}, Speed_dict: {Speed}\n\n")  


        #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), detected_color, box_width)
        # if text[-4:] == "km/h":
        #     text_size, _ = cv2.getTextSize(text, font, font_size, font_thickness)
        #     text_w, text_h = text_size

        #     pos_x = max((xmin - 20),0)
        #     pos_y = max((ymin - 20),0)
        #     cv2.rectangle(frame, (pos_x, pos_y), (pos_x + text_w, pos_y + text_h), color = (0,0,0), thickness = -1)
        #     cv2.putText(frame, text, (pos_x, pos_y + text_h + font_size - 1), font, font_size, font_color, font_thickness)
    
    return frame

def get_tracker_info(outputs):
    n = len(outputs)
    tracker_info = []
    for i in range(n):
        xmin = outputs[i][0] 
        ymin = outputs[i][1] + 10 
        xmax = outputs[i][2] 
        ymax = outputs[i][3] 
        
        id = int(outputs[i][4])
        
        tracker_info.append((xmin, ymin, xmax, ymax, id))
    
    return tracker_info

def get_tracking_id(result,frame):
    
    test = result.xyxy[0]
    temp = test[test[:, 5] > 0]
    test2 = temp[temp[:,5] < 8] 
    xywhs = xyxy2xywh(test2[:, 0:4])
    confs = test2[:, 4]
    clss =  test2[:, 5]
    
    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
    return outputs
    
# def write_in_txt(id,box,speed,filename="result.txt"):
#     text = str(id) + " " + str(box) + " " + str(speed) 
#     with open(filename, 'a') as f:
#         f.write(text)
#         f.write("\n")

video = cv2.VideoCapture(video_path)
TOTAL_FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video.get(3))
frame_height = int(video.get(4))
   
size = (frame_width, frame_height)
print(size)  


# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
# output_video = cv2.VideoWriter(output_video, 
#                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
#                          FPS, size)



start = timeit.default_timer()
pbar = tqdm(total=TOTAL_FRAMES)
while True:
    
    success, frame = video.read()
    if success:
        pbar.update(1)
        
        if start_from_checkpoint:
            if FRAME_COUNT > checkpoint:
                start_from_checkpoint = False
            
            FRAME_COUNT += 1
            continue
                
            
        
        if if_skip_frame():
            #print("Inside: ", FRAME_COUNT)
            FRAME_COUNT += 1
            if FRAME_COUNT % 7200==0:
                create_report(data,FRAME_COUNT)
            continue

        #print(FRAME_COUNT)


        result = model(frame)
        outputs = get_tracking_id(result,frame)
        
        tracker_info = get_tracker_info(outputs)
        
        #cv2.polylines(frame,[np.array([area_processed[0], area_processed[1], area_processed[2], area_processed[3]], np.int32)], True, (15, 220, 18), 6)
        #cv2.polylines(frame,[np.array([area_processed[4], area_processed[5], area_processed[6], area_processed[7]], np.int32)], True, (15, 220, 18), 6)
        frame = process_frame_for_speed(frame, tracker_info)
        
        # cv2.imshow("asda", frame)
        #output_video.write(frame)
        
        
        
        FRAME_COUNT = FRAME_COUNT + 1
        if FRAME_COUNT % 7200==0:
            create_report(data,FRAME_COUNT)
             
            
        # if frame_count == frame_skip:
        #     frame_count = 0
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    else:
        break


create_report(data,FRAME_COUNT+1)
create_log(log)
end = timeit.default_timer()
print(f"For processing one {TOTAL_FRAMES/FPS}s video: Total Required Time: {end-start}s")

