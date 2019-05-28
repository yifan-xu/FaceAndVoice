from src.detector import detect_faces
from src.utils import show_bboxes
from PIL import Image
import os

def main():
    for filename in os.listdir('video'):
        print(filename)
        #image = Image.open('images/test2.jpg')
        image = Image.open('video/' + filename)
        bounding_boxes, landmarks = detect_faces(image)
        image = show_bboxes(image, bounding_boxes, landmarks)
        image.save('output/'+filename)

if __name__ == "__main__":
    main()
