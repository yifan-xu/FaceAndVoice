#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from mtcnn.mtcnn import MTCNN
import json
import os

detector = MTCNN()
data = {}
for filename in os.listdir('../video2'):
    print(filename)
    image = cv2.imread('../video2/' + filename)
    results = detector.detect_faces(image)
    img_data = {}
    for idx, face in enumerate(results):
        bounding_box = face['box']
        keypoints = face['keypoints']
        img_data[idx] = {'box': [bounding_box[0], bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]]}# , 'keypoints': keypoints
    data[filename] = img_data
    # cv2.imwrite("../output2/" + filename, image)
json.dump(data, open("../out.json","w"))

