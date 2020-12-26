import csv
import numpy as np
from sympy.geometry import Polygon

def ommit_escape(txt):
    return txt[1:-1]

def exrtract_dimensions(polygons):
    xs = []
    ys = []

    for cordinate in polygons:
        xs.append(float(cordinate[0]))
        ys.append(float(cordinate[1]))

    return [min(xs), min(ys), max(xs), max(ys)]

def extract_segmentations(polygons):
    seg = []
    for cordinate in polygons:
        seg.append(float(cordinate[0]))
        seg.append(float(cordinate[1]))
    return seg

def calculate_area(polygons):
    cordinates = []
    for cordinate in polygons:
        t = (float(cordinate[0]),float(cordinate[1]))
        cordinates.append(t)

    area = np.absolute(float(Polygon(*tuple(cordinates)).area))
    return area

def return_objectList():
    with open('greenery/items_to_train.csv') as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        return l[0]