from flask import Blueprint, request, jsonify, redirect, url_for, render_template
from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import pandas as pd

from paddleocr import PaddleOCR
from werkzeug.utils import secure_filename
import os
import uuid
import logging
from flask import session
# from .imgPro import ImageProcess

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

# TODO IMG_PROCESSR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'reports/report')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'csv'}

# ocr_engine = PaddleOCR()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    chat_text = model.encode(request.form['chat'])
    query_result = collections.query(query_embeddings=[chat_text.tolist()], n_results=3)

    return query_result['metadatas'][0][0]['A']


@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # 'images[]'는 input 태그의 name 속성입니다.
        uploaded_files = request.files.getlist('images[]')
        
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                
                if filename.rsplit('.', 1)[1].lower() == 'csv':
                    df = pd.read_csv(filepath)
                    session['csv_filepath'] = filepath  # 세션에 파일 경로 저장
                    return jsonify({"message": "Upload successful"})
                else:
                    # 이미지 파일 처리 로직
                    # 여기에 추가적인 이미지 파일 처리 로직을 넣을 수 있습니다.
                    pass

        # 처리가 끝나면 map 라우트로 리다이렉트
        return redirect(url_for('mymap/map'))

    return render_template('map.html')  # GET 요청시 home.html을 렌더링


# #TODO YOLO & OCR & GPT
# @bp.route('/ocr', methods=['GET', 'POST'])
# def ocr():
#     if 'file' not in request.files:
#         return "No file part", 400

#     file = request.files['file']

#     if file.filename == '':
#         return "No selected file", 400

#     if not allowed_file(file.filename):
#         return "Invalid image format", 400

#     # 고유한 파일명 생성
#     ext = os.path.splitext(file.filename)[1]
#     unique_filename = str(uuid.uuid4()) + ext
#     filepath = os.path.join(UPLOAD_FOLDER, unique_filename)

#     file.save(filepath)
    
#     # run_all 함수 호출
#     
