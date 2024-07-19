import cv2
import numpy as np
from pathlib import Path
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type 
import openvino.runtime as ov

import time

SCORE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.7

def resize_and_pad(image, new_shape):
    old_size = image.shape[:2]
    ratio = float(new_shape[-1] / max(old_size))
    new_size = tuple([int(x * ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = new_shape[1] - new_size[1]
    delta_h = new_shape[0] - new_size[0]

    color = [100, 100, 100]
    new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)

    return new_im, delta_w, delta_h

def main():
    core = ov.Core()
    model = core.read_model(str(Path("/home/spr/robocon2024/Utils/yolov5_master/DATASET/weights/v5n_openvino_model/v5n.xml")))

    ppp = PrePostProcessor(model)
    ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
    ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale([255., 255., 255.])
    ppp.input().model().set_layout(Layout("NCHW"))
    ppp.output().tensor().set_element_type(Type.f32)
    model = ppp.build()
    compiled_model = core.compile_model(model, "GPU")

    cap = cv2.VideoCapture(0)  # Specify the video file path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        img_resized, dw, dh = resize_and_pad(frame, (640, 640))
        input_tensor = np.expand_dims(img_resized, 0)

        infer_request = compiled_model.create_infer_request()
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
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    x, y, w, h = prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item()
                    xmin = x - (w / 2)
                    ymin = y - (h / 2)
                    box = np.array([xmin, ymin, w, h])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

        detections = []
        for i in indexes:
            j = i.item()
            detections.append({"class_index": class_ids[j], "confidence": confidences[j], "box": boxes[j]})

        for detection in detections:
            box = detection["box"]
            class_id = detection["class_index"]
            confidence = detection["confidence"]

            rx = frame.shape[1] / (img_resized.shape[1] - dw)
            ry = frame.shape[0] / (img_resized.shape[0] - dh)
            box[0] = rx * box[0]
            box[1] = ry * box[1]
            box[2] = rx * box[2]
            box[3] = ry * box[3]

            xmax = box[0] + box[2]
            ymax = box[1] + box[3]
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(xmax), int(ymax)), (0, 255, 0), 3)
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1]) - 20), (int(xmax), int(box[1])), (0, 255, 0),
                                  cv2.FILLED)
            text = f"{class_id} ({confidence:.2f})"
            frame = cv2.putText(frame, text, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                 (0, 0, 0))

        end = time.time()

        print(1/(end - start))

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
