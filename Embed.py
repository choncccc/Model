import tensorflow as tf
import numpy as np
import torch
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# Load the pretrained model from facenet_pytorch
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')


def detect_and_align_faces(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)
    faces = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face = img_rgb[y1:y2, x1:x2]
            if face.size != 0:  
                face = cv2.resize(face, (160, 160))
                faces.append(face)

    return faces, img_rgb, boxes


def get_face_embedding(face):
    face = face / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = torch.tensor(face, dtype=torch.float32).unsqueeze(0)
    embedding = model(face).detach().cpu().numpy()
    return embedding


def texture_analysis(face):
    gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    return laplacian_var


def is_live(embedding, known_embedding, texture_score, threshold=0.6, texture_threshold=100.0):
    similarity = cosine_similarity(embedding, known_embedding)
    return similarity[0][0] > threshold and texture_score > texture_threshold


live_face_path = 'C:/Users/ULPI_OJT/img.jpg'
live_face_img = cv2.imread(live_face_path)
known_live_faces, _, _ = detect_and_align_faces(live_face_img)
if len(known_live_faces) > 0:
    known_live_face_embedding = get_face_embedding(known_live_faces[0])
else:
    print("Error: No face detected in the known live face image.")
    exit()


# Convert the model to TensorFlow and then to TensorFlow Lite
def convert_to_tflite():
    # Step 1: Convert to TensorFlow SavedModel
    dummy_input = tf.random.normal([1, 160, 160, 3])  # Example input shape for FaceNet

    class InferenceModel(tf.keras.Model):
        def __init__(self):
            super(InferenceModel, self).__init__()
            self.resnet = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(160, 160, 3)),
                                               tf.keras.layers.Lambda(lambda x: x / 255.0),
                                               tf.keras.layers.Conv2D(32, 3, activation='relu'),
                                               tf.keras.layers.GlobalAveragePooling2D()])
            
        def call(self, inputs):
            return self.resnet(inputs)

    model = InferenceModel()
    model(dummy_input)
    
    # Step 2: Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('facenet_model.tflite', 'wb') as f:
        f.write(tflite_model)


# Main function to run the face recognition using TensorFlow Lite model
def main():
    convert_to_tflite()  # Convert to TensorFlow Lite
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces, img_rgb, boxes = detect_and_align_faces(frame)
        
        if boxes is not None:
            embeddings = [get_face_embedding(face) for face in faces if face.size != 0]
            texture_scores = [texture_analysis(face) for face in faces]
            results = [is_live(embedding, known_live_face_embedding, texture_score)
                       for embedding, texture_score in zip(embeddings, texture_scores)]
    
            for i, (box, face) in enumerate(zip(boxes, faces)):
                if face.size != 0 and i < len(results):
                    x1, y1, x2, y2 = [int(b) for b in box]
                    color = (0, 255, 0) if results[i] else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = "Live" if results[i] else "Spoof"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, color, 2)
        
        cv2.imshow("Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
