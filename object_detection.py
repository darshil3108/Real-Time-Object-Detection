import cv2
import numpy as np

# Load SSD MobileNet v2 model (COCO dataset)
net = cv2.dnn.readNetFromTensorflow(
    "frozen_inference_graph.pb",
    "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
)

# Use CPU (set to DNN_TARGET_CUDA if you have GPU support)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ✅ Allowed 50 COCO class IDs mapped to human-readable names
allowed_ids = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorbike",
    6: "bus",
    8: "truck",
    10: "traffic light",
    13: "stop sign",
    15: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    21: "cow",
    44: "bottle",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    55: "orange",
    57: "carrot",
    58: "pizza",
    61: "cake",
    62: "chair",
    63: "sofa",
    64: "pottedplant",
    67: "diningtable",
    72: "tvmonitor",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Preprocess frame for network
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Process detections
    for detection in detections[0, 0, :, :]:
        confidence = detection[2]
        class_id = int(detection[1])

        if confidence > 0.5 and class_id in allowed_ids:
            box_x = int(detection[3] * w)
            box_y = int(detection[4] * h)
            box_width = int(detection[5] * w - box_x)
            box_height = int(detection[6] * h - box_y)

            # Draw bounding box
            cv2.rectangle(frame, (box_x, box_y),
                          (box_x + box_width, box_y + box_height),
                          (0, 255, 0), 2)

            label = f"{allowed_ids[class_id]}: {confidence*100:.1f}%"
            cv2.putText(frame, label, (box_x, box_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow("SSD MobileNetV2 - 50 Objects", frame)

    # Exit conditions (ESC, Q, or window close)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or Q
        break
    if cv2.getWindowProperty("SSD MobileNetV2 - 50 Objects", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
