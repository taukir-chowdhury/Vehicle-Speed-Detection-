import os

start_from_checkpoint = False
checkpoint = 0

result_path = "../results"
am_pm = 'am'
TIME_START = '9'
input_name = "demo"
#$video_path = f"../vehicle_data/{input_name}.m4v"
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

print(checkpoint)        
    
 