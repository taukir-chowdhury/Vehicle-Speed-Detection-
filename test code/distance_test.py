import numpy as np
from torch import rand
import cv2
import numpy as np
import random
import timeit 

start = timeit.default_timer()
polygon = np.array([(0,0), (0,5), (5,5),(5,0)])
limit = 400000
#print(polygon)
#pt = [(x,random.randint(5,300)) for x in range(4000)]
pt = [(x,y) for x,y in zip(range(limit), range(limit))]
print(pt[0])


def get_distance(center, line):
    p3 = center
    p1 = line[0]
    p2 = line[1]
    d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
    return d


line = np.array([(-50,-12), (0,5)])
# dist = get_distance(pt[30],line)
# # print("point is ", pt[30])
# # print("Line: Y axis")
# #dist = cv2.pointPolygonTest(polygon,pt[0],True)

# print(dist)
# end = timeit.default_timer()
# print((end-start)*1000)

manual = []
lib = []
t1 = 0
t2 = 0
for i in range(limit):
    start_manual = timeit.default_timer()
    d = get_distance(pt[i], line)
    end_manual= timeit.default_timer()
    tm = end_manual-start_manual
    t1 += tm
        
    start_library = timeit.default_timer()
    d2 = cv2.pointPolygonTest(polygon,pt[i],True)
    end_library = timeit.default_timer()
    tl = end_library-start_library
    t2 += tl
    
    if tm > tl:
        lib.append(tl)
    elif tm < tl:
        manual.append(tm)

print("Total Time for manual:",t1)
print("Total Time for lib:",t2)
if len(lib) > len(manual):
    print("Lib is better", len(lib))
else:
    print("Manual is better")
    
print((tm-tl)/tm * 100, "%")

    
    

