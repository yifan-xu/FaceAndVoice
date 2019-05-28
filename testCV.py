import cv2
import numpy
import json

frame = None
xl, yu, xr, yd = 0, 0, 0, 0

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY, frame, xl, yu, xr, yd
    if event == cv2.EVENT_LBUTTONDBLCLK:
        old_frame = frame.copy()
        cv2.rectangle(frame, (xl, yu), (xr, yd), (0,155,255), -1) # -1 to fill the rectangle
        frame = cv2.addWeighted(old_frame,0.7,frame,0.3,0, frame)
        cv2.imshow('image',frame)
        print('x = %d, y = %d'%(x, y))
        mouseX,mouseY = x,y

def main():
    global frame, xl, yu, xr, yd
    cap = cv2.VideoCapture("./video2/%03d.png",cv2.CAP_IMAGES)
    cv2.namedWindow("image")
    cv2.setMouseCallback('image',draw_circle)

    # cap = cv2.VideoCapture("./mtcnn/ivan_drawn.jpg",cv2.CAP_IMAGES)
    data = json.load(open("./out.json","r"))
    count = 1

    while(1):
        ret,frame = cap.read()
        if frame is None:
            break
        filename = "%03d.png" % count
        faces = data[filename]
        for idx, box in faces.items(): # for box in faces:
            xl, yu, xr, yd = box['box']
            cv2.rectangle(frame, (xl, yu), (xr, yd), (0,155,255), 2)
        
        cv2.imshow('image',frame)
        res = cv2.waitKey()
        if res == ord('q'):
           break;
        count += 1

if __name__ == "__main__":
    main()
