import numpy as np
import cv2
def detectObjects(img_path):
    confidenceThreshold = 0.5    #creating a threshold (a minimum limitation)
    NMSThreshold = 0.3           

    modelConfiguration = 'cfg/yolov3.cfg' #configuration file it consist of Deep learning architecture that yolo follows path
    modelWeights = 'yolov3.weights' #pretrained weights path

    labelsPath = 'coco.names' #this files consist of all pretrained classes path
    labels = open(labelsPath).read().strip().split('\n') #extracting data of coco files
    

    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights) #loading pretrained weights and configurations files 

    image = cv2.imread(img_path) #image reading
    (H, W) = image.shape[:2] #taking height and weight of input image

    #Determine output layer names
    layerName = net.getLayerNames()  #it will return layer name of module
    layerName = [layerName[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False) #converting into input format that yolo required (basically it contain 3 image i.e B G R image with a good image quality)
    net.setInput(blob) #giving input to pretrained module
    layersOutputs = net.forward(layerName) #here we are forwarding layer into into levels  and getting output from output layer(last layer)

    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs: #getting outputs from last layer i.e output layer
        for detection in output: #iterating to all detected objects
            scores = detection[5:] #basically detection is a list(not confirmed) that consist of 85 elements 1st consist of X cordinate, 2nd consist of Y cordinate, 3rd consist of  Height, 4th consist of width, 5th consist of score about the rounded box contain object or not and from 6 onwards it conist of confidence score about 1 to 80 class 
            classID = np.argmax(scores) # it will return the index pf highest score of class from score variable 
            confidence = scores[classID] # it will return the highest score 
            if confidence > confidenceThreshold: #comparing 
                box = detection[0:4] * np.array([W, H, W, H])  # here were multiplying the elements of detection list with np.array elements because to unnormalized the x,y  height,width
                (centerX, centerY,  width, height) = box.astype('int') #here will get centerX, centerY,  width, height after unnormalizing
                x = int(centerX - (width/2)) #now were calculating x y
                y = int(centerY - (height/2)) #now were calculating y

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    for ind, val in enumerate(boxes):
        print("index=",ind, val)
    for ind, val in enumerate(labels):
        print("org labels index=",ind, val)
   
    
    #Apply Non Maxima Suppression
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold) # it will removing the repeated box for same object and it will do this by keeping the box which has highest score    print(detectionNMS)
    outputs={}                          # from here onwards code are wriien for creating dictionary
    if(len(detectionNMS) > 0):                     
        outputs['detections']={}
        outputs['detections']['labels']=[]
        for i in detectionNMS.flatten():

                
            detection={}
            detection['label']=labels[classIDs[i]]
            detection['confidence']=confidences[i]
            detection['X']=boxes[i][0]
            detection['Y']=boxes[i][1]


            detection['Width']=boxes[i][2]
            detection['Height']=boxes[i][3]
            outputs['detections']['labels'].append(detection)
    return outputs
                



        # for i in detectionNMS.flatten():
        #     (x, y) = (boxes[i][0], boxes[i][1])
        #     (w, h) = (boxes[i][2], boxes[i][3])

        #     color = [int(c) for c in COLORS[classIDs[i]]]
        #     cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        #     text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
        #     cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Image', image)
    cv2.waitKey(0)



















# detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
# labels={}
# for i in detectionNMS:
#     outputs[labels]={}
#     labels={}
#     labels['X']:boxes[i][0],
#     labels['Y']:boxes[i][1],
#     labels['width']:boxes[i][2],
#     labels['height']:boxes[i][3],
#     labels['label']:labels[i][0],
#     labels['confidence']:confidences[i]
#     lab.append(labels)

    