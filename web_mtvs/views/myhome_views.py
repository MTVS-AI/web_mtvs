from flask import Blueprint, request, jsonify, redirect, url_for, render_template
from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import pandas as pd
import time

from paddleocr import PaddleOCR
from werkzeug.utils import secure_filename
import os
import uuid
import logging
from flask import session

from .imgPro import ImageProcess as IP
# from .views.folium_map import MapManager as MM
from .folium_class_4 import MapManager as MM

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

#TODO IMAGE PROCESS
def file_validation(filename):
    ALLOWED_EXTENSIONS = set(['jpg','JPG','png','PNG','mp4','json'])
    file_extension = filename.rsplit('.', 1)[1]
    return '.' in filename and file_extension in ALLOWED_EXTENSIONS , filename

@bp.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    chat_text = model.encode(request.form['chat'])
    query_result = collections.query(query_embeddings=[chat_text.tolist()], n_results=3)

    return query_result['metadatas'][0][0]['A']



def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['jpg','JPG','png','PNG','mp4','json'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS, filename

@bp.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('file[]')
        print(files)
        for fil in files:
            filename = secure_filename(fil.filename)
            print(fil)
            print(filename)
            time.sleep(0.01)
            result,filename  = allowed_file(filename)
            if result:
                fil.save(os.path.join('./web_mtvs/views/capture_data', filename))

        # 추가 ////
        aaa = IP('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','https://fsjr0lq9ke.apigw.ntruss.com/custom/v1/24396/82f04b3aebc287bf6b01f1571df49417fd2b38cb145fa7f9aadbb152eacbb606/general','cUZSQ3B0ZHpLZk53Q1JpeFpqQXFjd1VleGtZSW5keEY=')
        df_report = aaa.run_all(aaa.imgs, aaa.json_file_path)

        bbb = MM(df_report)
        bbb.create_folder(df_report)
        print("ran")

        return render_template('Map.html')  # GET 요청시 home.html을 렌더링