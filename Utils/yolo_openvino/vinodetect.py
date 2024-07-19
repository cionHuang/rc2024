import cv2
import numpy as np
from pathlib import Path
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type 
import openvino.runtime as ov

import time
# import numba 

SCORE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.65

def time_monitor(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper

class vinodetect:

    def __init__(self,model_path,device_name):
        
        self.core = ov.Core()
        self.model = self.core.read_model(str(Path(model_path)))

        self.ppp = PrePostProcessor(self.model)
        self.ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
        self.ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale([255., 255., 255.])
        self.ppp.input().model().set_layout(Layout("NCHW"))
        self.ppp.output().tensor().set_element_type(Type.f32)
        self.model = self.ppp.build()
        self.compiled_model = self.core.compile_model(self.model, device_name)

    def resize_and_pad(self,image, new_shape):
        old_size = image.shape[:2]
        ratio = float(new_shape[-1] / max(old_size))
        new_size = tuple([int(x * ratio) for x in old_size])

        image = cv2.resize(image, (new_size[1], new_size[0]))

        delta_w = new_shape[1] - new_size[1]
        delta_h = new_shape[0] - new_size[0]

        color = [100, 100, 100]
        new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)

        return new_im, delta_w, delta_h

    @time_monitor
    # @numba.njit
    def main(self,frame):
        
        # start = time.time()

        # 存放所有识别结果，为判断作准备
        self.red = []
        self.purple = []
        self.blue = []
        self.basket = []


        img_resized, dw, dh = self.resize_and_pad(frame, (640, 640))
        input_tensor = np.expand_dims(img_resized, 0)

        infer_request = self.compiled_model.create_infer_request()
        infer_request.infer({0: input_tensor})

        output = infer_request.get_output_tensor()
        detections = output.data[0]

        boxes = []
        class_ids = []
        confidences = []

        for prediction in detections:
            confidence = prediction[4].item()
            if confidence >= CONFIDENCE_THRESHOLD:
                classes_scores = prediction[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if classes_scores[class_id] > .25:
                    
                    x, y, w, h = prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item()
                    xmin = x - (w / 2)
                    ymin = y - (h / 2)
                    box = np.array([xmin, ymin, w, h])
                    
                    if (w*h) > 500 : # 当识别框大于一定值时才考虑为有效识别，后续可修改值####################
                        boxes.append(box)
                        confidences.append(confidence)
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

        detections = []
        for i in indexes:
            j = i.item()
            detections.append({"class_index": class_ids[j], "confidence": confidences[j], "box": boxes[j]})
            # print(boxes[j])

        for detection in detections:
            box = detection["box"]
            class_id = detection["class_index"]
            confidence = detection["confidence"]
            # print(class_id)
            # rx = frame.shape[1] / (img_resized.shape[1] - dw)
            # ry = frame.shape[0] / (img_resized.shape[0] - dh)
            # box[0] = rx * box[0]
            # box[1] = ry * box[1]
            # box[2] = rx * box[2]
            # box[3] = ry * box[3]

            xmax = box[0] + box[2]
            ymax = box[1] + box[3]
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(xmax), int(ymax)), (0, 255, 0), 3)
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1]) - 20), (int(xmax), int(box[1])), (0, 255, 0),
                                cv2.FILLED)
            text = f"{class_id} ({confidence:.2f})"
            frame = cv2.putText(frame, text, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0))
            
            # ROI区域线
            cv2.line(frame, (319, 0), (319, 479), (0, 165, 255), 1)
            cv2.line(frame, (321, 0), (321, 479), (0, 165, 255), 1)
            # red 0
            # blue 1 
            # purple 2 
            # basket 3

            # 存放识别信息

            if class_id == 0:
                self.red.append(box)
            if class_id == 1:
                self.blue.append(box)
            if class_id == 2:
                self.purple.append(box)
            if class_id == 3:
                self.basket.append(box)
            
            
        # print(self.red)
    

        # end = time.time()

        # # print(1/(end - start))
        # frame = cv2.putText(frame , str(1/(end - start)),(50,50),cv2.FONT_HERSHEY_SIMPLEX, 2,
        #                         (0, 0, 0))
        # cv2.imshow("Frame", frame)
        # cv2.waitKey(1)
    

