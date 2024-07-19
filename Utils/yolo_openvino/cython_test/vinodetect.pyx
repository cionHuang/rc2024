# Example of how the C++ code might be structured in a Cython-like syntax

cdef extern from "<opencv2/dnn.hpp>" namespace "cv":
    pass

cdef extern from "<openvino/openvino.hpp>" namespace "ov":
    pass

cdef extern from "<opencv2/opencv.hpp>" namespace "cv":
    pass

cdef float SCORE_THRESHOLD = 0.2
cdef float NMS_THRESHOLD = 0.4
cdef float CONFIDENCE_THRESHOLD = 0.4

cdef struct Detection:
    int class_id
    float confidence
    cv::Rect box

cdef struct Resize:
    cv::Mat resized_image
    int dw
    int dh

cdef Resize resize_and_pad(cv::Mat& img, cv::Size new_shape):
    # Implementation...

cdef void main():
    # Step 1. Initialize OpenVINO Runtime core
    cdef ov::Core core
    # Step 2. Read a model
    cdef std.shared_ptr[ov::Model] model = core.read_model("../../model/yolov5n.xml")

    # Step 3. Read input image
    cdef cv::Mat img = cv::imread("../../imgs/000000000312.jpg")
    # resize image
    cdef Resize res = resize_and_pad(img, cv::Size(640, 640))

    # Step 4. Inizialize Preprocessing for the model
    cdef ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model)
    # Specify input image format
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR)
    # Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.})
    # Specify model's input layout
    ppp.input().model().set_layout("NCHW")
    # Specify output results format
    ppp.output().tensor().set_element_type(ov::element::f32)
    # Embed above steps in the graph
    model = ppp.build()
    cdef ov::CompiledModel compiled_model = core.compile_model(model, "CPU")

    # Step 5. Create tensor from image
    cdef float *input_data = <float *> res.resized_image.data
    cdef ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data)

    # Step 6. Create an infer request for model inference 
    cdef ov::InferRequest infer_request = compiled_model.create_infer_request()
    infer_request.set_input_tensor(input_tensor)
    infer_request.infer()

    #Step 7. Retrieve inference results 
    cdef const ov::Tensor &output_tensor = infer_request.get_output_tensor()
    cdef ov::Shape output_shape = output_tensor.get_shape()
    cdef float *detections = output_tensor.data<float>()

    # Step 8. Postprocessing including NMS  
    cdef std.vector[cv::Rect] boxes
    cdef std.vector[int] class_ids
    cdef std.vector[float] confidences

    for i in range(output_shape[1]):
        # Postprocessing logic...

    cdef std.vector[int] nms_result
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result)
    cdef std.vector[Detection] output
    for i in range(nms_result.size()):
        # Postprocessing logic...

    # Step 9. Print results and save Figure with detections
    for i in range(output.size()):
        # Print and draw detections...

    cv::imwrite("./detection_cpp.png", img)

# Note: This is a simplified example for illustrative purposes and may require further adjustments.
