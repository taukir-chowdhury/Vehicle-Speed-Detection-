# import Point, Polygon
import timeit
from sympy import Point, Polygon, Line

start = timeit.default_timer()  
# creating points using Point()
#p1, p2, p3, p4 = map(Point, [(0, 0), (5, 0), (5, 5), (0, 5)])
p1, p2, p3, p4 = (0, 0), (5, 0), (5, 5), (0, 5)
p9, p10, p11, p12 = (3, 3), (3, -2), (-2, -2), (-2, 3)
p5, p6 = map(Point, [(-2,2), (6,3)])
# creating polygons using Polygon()
poly1 = Polygon(p1, p2, p3, p4)
poly2 = Polygon(p9, p10, p11, p12)
line1 = Line(p5, p6)
  
# using intersection()


l3 = Line((-2,2), (-6,3))
isIntersection = poly1.intersection(poly2)
print(isIntersection)
end = timeit.default_timer()
print(end-start)
