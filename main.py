import cv2



cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4,1080)

class_names = []
class_file = 'coco.names'

with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=0.5)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(255,0,0),thickness=3)
            cv2.putText(img, class_names[classId-1].upper(), (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 2,(255,0,0,2))

    cv2.imshow("Output", img)
    cv2.waitKey(0)