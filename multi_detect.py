from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

directory = './images/'

for file in os.listdir(directory):
    if file.endswith(".jpg") or file.endswith(".png"):
        path = (directory + file)
        detections = detector.detectObjectsFromImage(input_image=path, output_image_path="./results/" + file,
                                                     minimum_percentage_probability=50)

        for eachObject in detections:
            print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
            print("--------------------------------")
        continue
