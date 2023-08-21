from flask import Blueprint, request, render_template_string
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import folium
import logging
from flask import session
from .folium_map import MapManager


logging.basicConfig(level=logging.DEBUG)
bp = Blueprint('map', __name__, url_prefix='/mymap') 

#TODO CHATBOT
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
model = model.to('cpu')

client = chromadb.Client()
collections = client.create_collection('chatbot_new')

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

@bp.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    chat_text = model.encode(request.form['chat'])
    query_result = collections.query(query_embeddings=[chat_text.tolist()], n_results=3)

    return query_result['metadatas'][0][0]['A']


@bp.route('/map', methods=['GET', 'POST'])
def map():
    # MapManager 객체 생성
    csv_filepath = session.get('csv_filepath', 'reports/report/test_report_2023_08_12.csv')  # 세션에서 파일 경로 가져오기
    # map_manager = MapManager('reports/test_report_2023_08_12.csv')  # 적절한 파일 경로

    if csv_filepath:
        map_manager = MapManager(csv_filepath)
    else:
        print('csv_file not found')
    # folium.Map 객체 생성
    center = eval(map_manager.df.iloc[0]['Location'])  # 지도의 중심좌표
    map = folium.Map(location=center, zoom_start=12)
    # locations = ['[37.3953946671528, 127.1093297131001]', '[37.39417823717656, 127.1093844115068]', '[37.39271861487178, 127.10934841208203]', '[37.39185066864326, 127.11250910446297]', '[37.39345442116072, 127.11257925899302]', '[37.395544745716585, 127.11262755830957]', '[37.39650119524654, 127.111183435904]', '[37.39649911490269, 127.11337434990375]', '[37.39601079572209, 127.11520313115568]', '[37.39440567382231, 127.11653325709659]', '[37.39346684071838, 127.11831608687358]', '[37.39080112282156, 127.11700196362115]', '[37.39188034187261, 127.119013740367]', '[37.3940362949143, 127.10695620640082]', '[37.396431672499325, 127.10842774919402]', '[37.397943736734206, 127.11021431692127]']
    # df['Location'] = locations
    # # 각 위치에 대한 Marker와 Popup 추가
    for i in range(len(map_manager.df)):
        popup = map_manager.get_folium_popups(i)
        folium.Marker(
            location=eval(map_manager.df.iloc[i]['Location']), 
            icon=folium.Icon(color='blue', icon='star'), 
            popup=popup
        ).add_to(map)

    # 지도를 HTML 문자열로 변환
    map_html = map._repr_html_()

    # 지도를 렌더링
    return render_template_string('<html><body>{{map_html|safe}}</body></html>', map_html=map_html)