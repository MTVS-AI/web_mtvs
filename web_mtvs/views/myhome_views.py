from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import pandas as pd

from paddleocr import PaddleOCR
from werkzeug.utils import secure_filename
import os
import uuid
import logging

logging.basicConfig(level=logging.DEBUG)
bp = Blueprint('myhome', __name__, url_prefix='/myhome') 

#TODO CHATBOT
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
model = model.to('cpu')

client = chromadb.Client()
collections = client.create_collection('chatbot')

df = pd.read_csv('ChatbotData.csv')
df1 = pd.read_csv('embeding.csv', header=None)
embeddings=[]
metadata = []
ids = []

for temp in range(len(df1)):
    ids.append(str(temp+1))  
    embeddings.append(df1.iloc[temp].tolist())
    metadata.append({'A' : df.iloc[temp]['A']})

collections.add(embeddings=embeddings,metadatas=metadata,ids=ids)

# TODO OCR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads/')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

ocr_engine = PaddleOCR()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    chat_text = model.encode(request.form['chat'])
    query_result = collections.query(query_embeddings=[chat_text.tolist()], n_results=3)

    return query_result['metadatas'][0][0]['A']


@bp.route('/ocr', methods=['GET', 'POST'])
def ocr():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "Invalid image format", 400

    # 고유한 파일명 생성
    ext = os.path.splitext(file.filename)[1]
    unique_filename = str(uuid.uuid4()) + ext
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)

    file.save(filepath)

    try:
        result = ocr_engine.ocr(filepath)
        logging.debug("OCR result: %s", result)

        # 텍스트만 추출하기 위한 코드
        for line in result:
            logging.debug("Type of line[1][0]: %s, Value: %s", type(line[1][0]), line[1][0])

        texts = [line[1][0] for line in result if len(line) > 1 and len(line[1]) > 0]
        extracted_text = ' '.join(texts)

    except Exception as e:
        logging.error("OCR 처리 중 오류 발생: ", exc_info=True)
        return f"OCR error: {str(e)}", 500

    # os.remove(filepath)  # OCR 처리된 이미지 파일 삭제 (필요하다면 주석 해제)
    return jsonify({"text": extracted_text})