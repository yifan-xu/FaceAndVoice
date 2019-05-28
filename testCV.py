import cv2
import numpy
import json

# frame = None
# xl, yu, xr, yd = 0, 0, 0, 0


def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY, frame, L, T, R, B 
    if event == cv2.EVENT_LBUTTONDBLCLK:
        old_frame = frame.copy()
        cv2.rectangle(frame, (L, T), (R, B), (0,155,255), -1) # -1 to fill the rectangle
        frame = cv2.addWeighted(old_frame,0.7,frame,0.3,0, frame)
        cv2.imshow('image',frame)
        print('x = %d, y = %d'%(x, y))
        mouseX,mouseY = x,y

def selectFace(face, frame, out):
    L, T, R, B = face['box']
    old_frame = frame.copy()
    cv2.rectangle(frame, (L, T), (R, B), (0,155,255), -1) # -1 to fill the rectangle
    frame = cv2.addWeighted(old_frame,0.7,frame,0.3,0, frame)
    cv2.imshow('image',frame)  
    while(1):
        res = cv2.waitKey()
        if res == 32:
            interval = max(B-T, R-L)
            sub_frame = frame[T:T+interval, L:L+interval]
            dst = cv2.resize(sub_frame, (200,200));
            out.write(dst)

            break  
        elif res == ord('q'):
            # cap.release()
            out.release()
            cv2.destroyAllWindows()
            exit()

# TODO: Crop the face as a square
def cropSquare():
    pass

def getOverlap(r1, r2):
    r1_L, r1_T, r1_R, r1_B = r1['box']
    r2_L, r2_T, r2_R, r2_B = r2['box']
    left = max(r1_L, r2_L)
    right = min(r1_R, r2_R)
    bottom = min(r1_B, r2_B)
    top = max(r1_T, r2_T)
    if left < right and bottom > top:
        intersection = getArea(left, right, bottom, top)
        r1_a = getArea(r1_L, r1_R, r1_B, r1_T)
        r2_a = getArea(r2_L, r2_R, r2_B, r2_T)
        overlap = intersection / (r1_a + r2_a - intersection)
    else:
        overlap = 0
    return overlap

def getArea(left, right, bottom, top):
    return (right - left) * (bottom - top)

def main():
    # global frame, L, T, R, B 
    prev_face = None
    cap = cv2.VideoCapture("./video2/%03d.png",cv2.CAP_IMAGES)
    # width = cv2.CAP_PROP_FRAME_WIDTH
    # print(width)
    # height = cv2.CAP_PROP_FRAME_HEIGHT
    # print(height)

    out = cv2.VideoWriter('./outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 1, (200,200))
    cv2.namedWindow("image")
    cv2.setMouseCallback('image',draw_circle)
    font = cv2.FONT_HERSHEY_SIMPLEX
    overlap_percent = 0.8

    # cap = cv2.VideoCapture("./mtcnn/ivan_drawn.jpg",cv2.CAP_IMAGES)
    data = json.load(open("./out.json","r"))
    count = 1

    while(1):
        ret,frame = cap.read()
        if frame is None:
            break
        filename = "%03d.png" % count
        faces = data[filename]
        overlap = 0
        max_idx = None
        for idx, box in faces.items(): # for box in faces:
            if prev_face is not None:
                new_overlap = getOverlap(box, prev_face)
                if new_overlap > overlap:
                    overlap = new_overlap
                    max_idx = idx
            L, T, R, B = box['box']
            cv2.rectangle(frame, (L, T), (R, B), (0,155,255), 2)
            cv2.putText(frame,str(idx),(L, T), font, 1,(255,255,255),2,cv2.LINE_AA)
        if overlap > overlap_percent:
            prev_face = faces[max_idx]
            selectFace(faces[max_idx], frame, out)
        else:
            cv2.imshow('image',frame)
            res = cv2.waitKey()

            if str(res-48) in faces.keys():
                prev_face = faces[str(res-48)]
                selectFace(faces[str(res-48)], frame, out)

            elif res == ord('q'):
                break
            else:
                prev_face = None

        count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
