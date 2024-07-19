import cv2
import numpy as np
from openvino.inference_engine import IECore

class Detector:
    def __init__(self):
        self.ie = IECore()
        self.net = None
        self.input_name = None
        self.cof_threshold = None
        self.nms_area_threshold = None

    def init(self, xml_path, cof_threshold, nms_area_threshold):
        self.net = self.ie.read_network(model=xml_path)
        self.input_name = next(iter(self.net.input_info))
        self.cof_threshold = cof_threshold
        self.nms_area_threshold = nms_area_threshold
        self.exec_net = self.ie.load_network(network=self.net, device_name="CPU")
        return True

    def uninit(self):
        self.exec_net = None
        self.net = None
        return True

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_anchors(self, net_grid):
        anchors = [24,31, 39,47, 50,71] if net_grid == 80 else \
                  [69,61, 70,155, 76,91] if net_grid == 40 else \
                  [99,76, 101,116, 122,171]
        return anchors

    def parse_yolov5(self, blob, net_grid, cof_threshold, o_rect, o_rect_cof):
        blob_data = blob.buffer
        item_size = 85
        anchor_n = 3
        for n in range(anchor_n):
            for i in range(net_grid):
                for j in range(net_grid):
                    box_prob = self.sigmoid(blob_data[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 4])
                    if box_prob < cof_threshold:
                        continue

                    x = blob_data[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 0]
                    y = blob_data[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 1]
                    w = blob_data[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 2]
                    h = blob_data[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 3]

                    max_prob = 0
                    idx = 0
                    for t in range(5, 85):
                        tp = blob_data[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ t]
                        tp = self.sigmoid(tp)
                        if tp > max_prob:
                            max_prob = tp
                            idx = t
                    cof = box_prob * max_prob
                    if cof < cof_threshold:
                        continue

                    x = (self.sigmoid(x)*2 - 0.5 + j) * 640.0 / net_grid
                    y = (self.sigmoid(y)*2 - 0.5 + i) * 640.0 / net_grid
                    w = pow(self.sigmoid(w)*2, 2) * self.get_anchors(net_grid)[n*2]
                    h = pow(self.sigmoid(h)*2, 2) * self.get_anchors(net_grid)[n*2 + 1]

                    r_x = x - w/2
                    r_y = y - h/2
                    rect = (round(r_x), round(r_y), round(w), round(h))
                    o_rect.append(rect)
                    o_rect_cof.append(cof)
        return True if o_rect else False

    def detet2origin(self, dete_rect, rate_to, top, left):
        pass

    def process_frame(self, inframe):
        if inframe is None:
            print("无效图片输入")
            return False
        inframe_resized = cv2.resize(inframe, (640, 640))
        inframe_rgb = cv2.cvtColor(inframe_resized, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(inframe_rgb.astype(np.float32), axis=0)
        self.exec_net.start_async(request_id=0, inputs={self.input_name: blob})
        if self.exec_net.requests[0].wait(-1) == 0:
            outputs = self.exec_net.requests[0].output_blobs
            origin_rect = []
            origin_rect_cof = []
            s = [80, 40, 20]
            i = 0
            for output_name in outputs.keys():
                blob = outputs[output_name]
                self.parse_yolov5(blob, s[i], self.cof_threshold, origin_rect, origin_rect_cof)
                i += 1

            final_id = cv2.dnn.NMSBoxes(origin_rect, origin_rect_cof, self.cof_threshold, self.nms_area_threshold)
            detected_objects = []
            for i in final_id:
                rect = origin_rect[i[0]]
                detected_objects.append({
                    "prob": origin_rect_cof[i[0]],
                    "name": "",
                    "rect": rect
                })
            return detected_objects
        else:
            return []

detector = Detector()
xml_path = "/home/huang/rc/Utils/yolov5_master/openvino_rc2024.xml"
detector.init(xml_path, 0.1, 0.5)

src = cv2.imread("/home/huang/rc/data/original/ball_frame/240207/image6.png")
detected_objects = detector.process_frame(src)
for obj in detected_objects:
    xmin, ymin, width, height = obj["rect"]
    cv2.rectangle(src, (xmin, ymin), (xmin+width, ymin+height), (0, 0, 255), 1)
cv2.imshow("result", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
