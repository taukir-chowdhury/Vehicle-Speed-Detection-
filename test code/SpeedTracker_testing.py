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
from tqdm import tqdm

### Input and Output Path ###
result_path = "F:\Work\Personal\\results"

input_name = "test_300feet2"
if not os.path.exists(os.path.join(result_path,input_name)):
    os.makedirs(os.path.join(result_path,input_name))
    os.makedirs(os.path.join(result_path,input_name,"saved_images"))
    
    
output_name = 'test_300feet2_output_2'
output_video = f"{result_path}\\{input_name}\{output_name}.avi"
video_path = f"F:\Work\Personal\\test\\{input_name}.mp4"
FPS = 12


### Model Loading ###
model = torch.hub.load("F:\Work\Personal\yolov5", "custom", 
                        source="local", path="F:\Work\Personal\yolov5\\yolov5s.pt", force_reload=True)

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

LENGTH = 16 * .001 #km or 16 meter

x_factor = x_numpy/x_editor
y_factor = y_numpy/y_editor

def process_coordinates(area):
    area_processed = []
    for co in area:
        area_processed.append((co[0] * x_factor, co[1] * y_factor ))
    
    return area_processed

#area = [(420,650), (645,365), (1372,370), (1550,650)]
# factors = np.array([[x_factor,0],[0,y_factor]])

# area = np.array(np.mat('3 870 ; 4.5 790 ; 373 711 ; 1047 718 ; 1192 916'))
# area_processed = np.matmul(area,factors)

# dr1 = np.array([[18,1021], [4,1066], [1340,1063], [1315,1023]])
# dr2 = np.array([[508,607], [657,501], [997,519], [1060,621]])
# deleting_region1 = np.matmul(dr1,factors)
# deleting_region2 = np.matmul(dr2,factors)
# #,[655,637]
# roi = np.array([[16,1068],[6,780],[969,639],[1341,1059]])
# roi_processed = np.matmul(roi,factors)


area = [(565,298), (1005,417), (861,633), (198,423)]
dr1 = [(0,470), (520,750), (417,900), (0,700)]
entry_region = [(565,298), (1005,417), (960,448), (526,319)]

deleting_region1 = process_coordinates(dr1)
area_processed = process_coordinates(area)


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

def calculate_speed(entry_time,exit_time):
    time_diff = exit_time - entry_time
    #print(exit_time,entry_time)
    speed = (LENGTH/time_diff) * 3600
    print(speed)
    return round(speed,3)



def adding_to_report(frame,ID,speed,time, xmin, ymin, xmax, ymax):
    
    #cropped = img[start_row:end_row, start_col:end_col]
    cropped_image = frame[ymin:ymax,xmin:xmax]
    name = f"{time*FPS}_{ID}"
    cv2.imwrite(f"{os.path.join(result_path,input_name,'saved_images')}\{name}.png", cropped_image)
    data.append([name,time,speed])

    
def create_report(data):
    header = ['name', 'time(s)', 'speed(km/h)']
    with open(f"{result_path}\\{input_name}\\report.csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)

def create_log(log):
    with open("f'{result_path}\\{input_name}\{output_name}_log.txt'", "w") as f:
        text = "".join(log)
        f.write(text)

def annotate(frame, obj_info,font=cv2.FONT_HERSHEY_SIMPLEX, box_width=1, font_size=1, 
            detected_color=(0, 255, 0), font_color=(255, 255, 255), font_thickness = 1):

    
    for obj in obj_info:
        #top, right, bottom, left
        xmin, ymin, xmax, ymax = obj[0], obj[1], obj[2], obj[3]
    
        
        ID = int(obj[-1])
        
        text = ""
        cx = int((xmin + xmax)/2)
        cy = int((ymin + ymax)/2)
        
        if Entered_Polygon.get(ID, -1) == -1 and not if_inside_polygon((cx, cy), area_processed):
            log.append(f"ID: {ID} centered: ({cx},{cy}) in not inside polygon and haven't entered yet\n")
            pass
        
        elif Entered_Polygon.get(ID, -1) == -1 and if_inside_polygon((cx, cy), area_processed):
            entry_time = FRAME_COUNT/FPS
            Entered_Polygon[ID] = entry_time
            log.append(f"ID: {ID} centered: ({cx},{cy}) inside polygon but not in Entered_polygon dict yet. Entry time: {entry_time}, frame count: {FRAME_COUNT}\n")
        
        elif Entered_Polygon.get(ID, -1) != -1 and not if_inside_polygon((cx, cy), area_processed):
            log.append(f"ID: {ID} centered: ({cx},{cy}) inside polygon and in Entered_polygon\n")
            if Speed.get(ID,-1) == -1 :
                exit_time = FRAME_COUNT/FPS
                entry_time = Entered_Polygon.get(ID)
                speed = calculate_speed(entry_time,exit_time)
                
                Speed[ID] = speed
                log.append(f"ID: {ID} centered: ({cx},{cy}) not in polygon but in Entered_polygon, so, calculating Speed. speed: {speed}, Entry Time: {entry_time}, Exit Time: {exit_time}, frame Count {FRAME_COUNT}\n")
                log.append(f"ID: {ID} centered: ({cx},{cy}) frame: {FRAME_COUNT}, Speed_dict: {Speed}\n\n")
                
                
            elif if_inside_polygon((cx, cy),deleting_region1):
                speed = Speed[ID]
                time = FRAME_COUNT/FPS
                log.append(f"ID: {ID} centered: ({cx},{cy})  inside deleting region with speed: {speed}.\n")
                log.append(f"Speed_dict before deleting inside deleting region: {Speed} \n\n")
                log.append(f"EnteredPolygon_dict before deleting inside deleting region: {Entered_Polygon} \n\n")
                adding_to_report(frame,ID,speed,time,xmin, ymin, xmax, ymax)
                del Speed[ID]
                del Entered_Polygon[ID]
                log.append(f"EnteredPolygon_dict after deleting inside deleting region: {Entered_Polygon} \n\n")
                log.append(f"Speed_dict After deleting inside deleting region: {Speed} \n\n")
            else:
                
                speed = Speed[ID]
                log.append(f"ID: {ID} centered: ({cx},{cy}) frame: {FRAME_COUNT}, with speed: {speed}\n\n")
                
            text = str(speed) + "km/h" 
            
        #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), detected_color, box_width)
        if text[-4:] == "km/h":
            text_size, _ = cv2.getTextSize(text, font, font_size, font_thickness)
            text_w, text_h = text_size

            pos_x = max((xmin - 20),0)
            pos_y = max((ymin - 20),0)
            cv2.rectangle(frame, (pos_x, pos_y), (pos_x + text_w, pos_y + text_h), color = (124,252,0), thickness = -1)
            cv2.putText(frame, text, (pos_x, pos_y + text_h + font_size - 1), font, font_size, font_color, font_thickness)
            log.append(f"ID: {ID} centered: ({cx},{cy}) inside puttext: frame: {FRAME_COUNT}, with speed: {text}\n\n")
    
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
output_video = cv2.VideoWriter(output_video, 
                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                         FPS, size)


frame_skip = 0
start = timeit.default_timer()

pbar = tqdm(total=TOTAL_FRAMES)
while True:
    
    success, frame = video.read()
    if success:
        pbar.update(1)
        result = model(frame)
        outputs = get_tracking_id(result,frame)
        
        tracker_info = get_tracker_info(outputs)
        
        # cv2.polylines(frame,[np.array(area_processed, np.int32)], True, (15, 220, 18), 6)
        # cv2.polylines(frame,[np.array(dr1, np.int32)], True, (15, 220, 18), 6)
        frame = annotate(frame, tracker_info)
        
        # cv2.imshow("asda", frame)
        output_video.write(frame)
        
        
        
        FRAME_COUNT = FRAME_COUNT + 1
            
        # if frame_count == frame_skip:
        #     frame_count = 0
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    else:
        break


create_report(data)
create_log(log)

end = timeit.default_timer()
print(f"For processing one {TOTAL_FRAMES/FPS}s video: Total Required Time: {end-start}s")


    
