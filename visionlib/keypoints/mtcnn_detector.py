from mtcnn import MTCNN
import cv2

class MTCNNDetector:
    def __init__(self):
        self.model = MTCNN()

    def detect(self, img, rects, enable_gpu):
        detected = self.model.detect_faces(img)
        faces, points = [], []
        for face in detected:
            for key in face.keys():
                if key == 'keypoints':
                    for value in face[key]:
                        faces.append(face[key][value])
                    points.append(faces)
        return points
