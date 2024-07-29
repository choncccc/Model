from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import io

app = Flask(__name__)

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
            if face.size != 0:  # Check if face is valid
                face = cv2.resize(face, (160, 160))
                faces.append(face)

    return faces, img_rgb, boxes

def get_face_embedding(face):
    face = face / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = torch.tensor(face, dtype=torch.float32).unsqueeze(0)
    embedding = model(face).detach().cpu().numpy()
    return embedding

def is_live(embedding, known_embedding, threshold=0.6):
    similarity = cosine_similarity(embedding, known_embedding)
    return similarity[0][0] > threshold

live_face_path = 'C:/Users/ULPI_OJT/Desktop/Model/img.jpg'
live_face_img = cv2.imread(live_face_path)
known_live_faces, _, _ = detect_and_align_faces(live_face_img)
known_live_face_embedding = get_face_embedding(known_live_faces[0])

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    faces, img_rgb, boxes = detect_and_align_faces(img)
    
    if boxes is not None:
        embeddings = [get_face_embedding(face) for face in faces]
        results = [is_live(embedding, known_live_face_embedding) for embedding in embeddings]
        response = [{'box': box.tolist(), 'is_live': result} for box, result in zip(boxes, results)]
    else:
        response = []

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
