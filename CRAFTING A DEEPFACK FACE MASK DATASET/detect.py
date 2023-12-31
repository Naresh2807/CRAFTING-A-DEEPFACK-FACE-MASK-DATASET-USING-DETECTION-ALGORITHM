from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2
from tensorflow.keras.applications import DenseNet121

# Constants
DETECT_PROTOTXT = './DNN_face_detector/deploy.prototxt'
DETECT_CAFFE_MODEL = './DNN_face_detector/res10_300x300_ssd_iter_140000.caffemodel'
CLASSIFIER_MODEL = './DF-DenseNet121.h5'
LABEL_ENCODER = './le.pickle'
CONFIDENCE_THRESHOLD = 0.5

def detect(imgpath, outname):
    try:
        # Load face detection model
        network = cv2.dnn.readNetFromCaffe(DETECT_PROTOTXT, DETECT_CAFFE_MODEL)

        # Load face classification model
        model = load_model(CLASSIFIER_MODEL)

        # Load label encoder
        le = pickle.loads(open(LABEL_ENCODER, "rb").read())

        # Read the image
        img = cv2.imread(imgpath)
        (h, w) = img.shape[:2]

        # Perform face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        network.setInput(blob)
        detections = network.forward()

        # Process each detected face
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Crop and preprocess the face for classification
                face = cv2.resize(img[startY:endY, startX:endX], (224, 224))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # Model prediction
                # Model prediction
                preds = model.predict(face)[0]
                max_pred_value = np.max(preds)
                j = int(np.argmax(preds))
                label = le.classes_[j]

# Draw bounding box and label on the image
                label_text = "{}: {:.4f}".format(label, max_pred_value)
                color = (0, 0, 255) if max_pred_value < 1 else (0, 255, 0)
                cv2.putText(img, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)


        # Save the output image
        out = cv2.imwrite(f"./uploads/{outname}.jpg", img)
        print("Output image saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Specify the image path and output name
    image_path = 'path/to/your/image.jpg'
    output_name = 'output_image'

    # Call the detect function with the specified parameters
    detect(image_path, output_name)
