📌 Real-Time Object Detection using MobileNetV2 SSD



This project demonstrates real-time object detection using OpenCV and TensorFlow’s MobileNetV2 SSD model, trained on the COCO dataset.

It detects 50+ commonly available objects (person, car, dog, bottle, etc.) from a live camera feed and displays bounding boxes with confidence scores.



🚀 Features



Real-time detection from webcam



Uses pre-trained MobileNetV2 SSD (COCO) model



Detects 50+ everyday objects with bounding boxes



Displays object name + confidence score



Press q to exit the program



🛠️ Technology Stack



Python 3.7+



OpenCV (cv2.dnn module)



TensorFlow (pre-trained model)



📂 Project Structure

object\_detection/

│── object\_detection.py       # Main detection script

│── frozen\_inference\_graph.pb # Pre-trained model (downloaded)

│── ssd\_mobilenet\_v2\_coco.pbtxt # Model config file

│── README.md                 # Project documentation



📦 Installation \& Setup



Clone the repo / Create project folder



git clone https://github.com/your-username/object-detection-mobilenet.git

cd object-detection-mobilenet





Install dependencies



pip install opencv-python

pip install tensorflow





Download Pre-trained Model



SSD MobileNetV2 COCO model



Extract and place files:



frozen\_inference\_graph.pb



ssd\_mobilenet\_v2\_coco.pbtxt



Run the script



python object\_detection.py



🖥️ Usage



The webcam starts automatically.



Detected objects will be highlighted with bounding boxes + labels.



Press q to quit.



🎯 Example Output

Person: 98.4%

Dog: 92.1%

Bottle: 87.5%





Bounding boxes appear around detected objects in real-time.



📌 Future Enhancements



Add support for custom object detection datasets



Integrate with Flask/Django for web-based detection



Save detection results as logs or video files



👨‍💻 Author



Developed by Darshil Vaghela

MSc IT | Data Science \& AI Enthusiast

