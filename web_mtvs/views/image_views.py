from ultralytics import YOLO
import torch
from datetime import datetime
import shutil
import os
from glob import glob
import time
import json
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm


json_file_path = 'imgpath'
with open(json_file_path, 'r') as json_file:
    meta_data = json.load(json_file)

model = YOLO('best.pt')
imgs = glob('capture_data/*.jpg')

def make_frame(meta_data):
    df_data = []
    for item in tqdm(meta_data['data']):
        date_time = item['timestamp'].split("T")
        df_row = {
            'ID' : item['id'],
            'Date' : date_time[0],
            'Time' : date_time[1],
            'Location' : [item['location']['latitude'], item['location']['longitude']],
            'Origin_img' : [np.array(Image.open('capture_data/'+item['file_name'])).tolist()],
            'Detect_img' : [],
            'Crop_classes' : [],
            'Crop_imgs' : [],
            'Crop_xyxy' : [], # 좌표 시작점 x,y / 끝점 x,y
            'Crop_conf': [],
            'PaddleOCR_text' : [], # PaddleOCR
            'ClovaOCR_text' : [], # Naver Clova OCR
            'Category' : [],
            'Category_basis' : [],
            'Legality' : []
        }

        df_data.append(df_row)

    df_report = pd.DataFrame(df_data)

    # date_created : 데이터 수집한 날짜
    date_created = meta_data['dataset_info']['date_created'].split("T")[0].split("-")
    df_report.to_csv('reports/report_'+'_'.join(date_created)+'.csv')
    return df_report


class_names = ['banner','frame']
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']

# 경로 에러 시 predict/, predict2/, predict3/... 등으로 변경

predict_path = 'runs/detect/predict/'
predict_detect_path = predict_path
predict_crop_banner_path = predict_path+class_names[0]
predict_crop_frame_path = predict_path+class_names[1]

log_path = 'logs/'


def move_all_img(source_folder, destination_folder):
    # source_folder 내의 모든 항목을 destination_folder로 이동
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder, item)
        destination_item = os.path.join(destination_folder, item)
        shutil.move(source_item, destination_item)

        def move_img(source_path, destination_path):
            if not os.path.exists(os.path.dirname('/'.join(destination_path.split('/')[:-1])+'/')):
                os.makedirs(os.path.dirname('/'.join(destination_path.split('/')[:-1])+'/'))
            
            # 파일 이동
            shutil.move(source_path, destination_path)

                                # image는 원본이미지
def check_category(df_report,id,image,crop_classes,crop_xyxy):
    
    print("@@@@@@@@@@@@@@@@@@@@@@@ id @@@@@@@@@@@@@@@@@@@@@ :",id)
    # print("원본 사이즈 : ",image.shape)

    # resize된 상태에서 xy좌표를 뽑기 때문에 다시 resize
    image = np.resize(image,(640,640))
    # print("원본 사이즈에서 리사이즈 : ",np.resize(image,(640,640)).shape)

    frameXYXY_list = []

    for idx,class_name in enumerate(crop_classes):
        if class_name=='frame':
            frameXYXY_list.append(crop_xyxy[idx])
    
    for i,frameXYXY in enumerate(frameXYXY_list):
        print(str(i+1)+"번째 frameXYXY :",frameXYXY)
        for j,cropXYXY in enumerate(crop_xyxy):
            if crop_classes[j]!='frame':
                print(str(i+1)+"번째 cropXYXY :",cropXYXY)

                frame_min_x, frame_min_y, frame_max_x, frame_max_y = frameXYXY
                crop_min_x, crop_min_y, crop_max_x, crop_max_y = cropXYXY

                tmp_origin_img = np.array(image.copy(),dtype=np.int16)
                tmp_origin_img[frame_min_x:frame_max_x,frame_min_y:frame_max_y] = -1

                target_region = tmp_origin_img[crop_min_x: crop_max_x, crop_min_y :crop_max_y]
                # print("target_region.size : ",target_region.size)
                negative_one_pixel_count = np.count_nonzero(target_region == -1)
                total_pixel_count = target_region.size
                negative_one_percentage = (negative_one_pixel_count / total_pixel_count) * 100

                print(str(i+1)+"번째 frame"+str(j+1)+"번째 crop이미지 겹치는 범위 : ",negative_one_percentage)

                # frame 안에 현수막이 70% 이상 포함되면 pulbic
                # -1:초기화, 0:프레임, 1:공익, 2:정치 3:기타 
                if negative_one_percentage >= 70:
                    df_report.iloc[id]['Category'][j] = 1
                    df_report.iloc[id]['Category_basis'][j] = 1

    return df_report


def yolo_run(img,df_report):

    # GPU predict 시, show=False/save=True로 설정 
    results = model.predict(
                            source=img, # 디렉토리 (capture_data/)
                            conf=0.5, # confidence threshold for detection (오탐지 시 재설정)
                            save=True,  # Detect 결과 저장 (runs/detect/predict)
                            device=device, # device 설정
                            show=False, # window 창으로 show
                            save_crop=True # Detect된 Obeject 사진 저장 (runs/detect/predict/crops)
                            )
    
    image = np.array(Image.open(img))

    for idx,result in enumerate(results):
        now = datetime.now()
        now_time = str(now.year) + str(now.month) + str(now.day) + '_' + str(now.hour) + str(now.minute) + str(now.second)

        boxes = result.boxes
        saved_img = ''

        print("================================= Predict 결과 =================================")

        file_name = img.split('\\')[-1]
        img_name = file_name[:-4]

        ### 데이터 logs에 저장
        # 데이터이름+날짜시간
        data_datetime_dir = log_path+now_time+'_'+img_name+'/'

        # 오류나서 추가1
        os.makedirs(data_datetime_dir)

        # predict된 결과를 'logs/'에 저장
        move_all_img(predict_path,data_datetime_dir)

        # 오류나서 추가2
        time.sleep(1)
        # 원본 이미지 저장
        os.makedirs(data_datetime_dir+'origin_img/')
        shutil.copyfile(img,data_datetime_dir+'origin_img/'+img.split('\\')[-1])
        # detect된 이미지 저장
        move_img(data_datetime_dir+file_name,data_datetime_dir+'detect_img/'+file_name)
        
        try:
            # crop_imgs로 폴더명 변경
            os.rename(data_datetime_dir+'crops', data_datetime_dir+'crop_imgs')
        except:
            print("Crop된 이미지 없음! (Detect이미지 없음!)")

        ### 데이터 df에 저장
        # ID 값
        id = int(img_name.split('_')[-1])-1
        # detect_img 기록
        df_report['Detect_img'].iloc[id] = np.array(Image.open(data_datetime_dir+'detect_img/'+file_name)).tolist()

        crop_classes = []
        crop_imgs = []
        crop_xyxy = []
        crop_conf = []
        for idx,box in enumerate(boxes):
            # crop된 object 이름 기록
            crop_classes.append(class_names[int(box.cls)])
            # crop된 xyxy 좌표 기록
            xyxy = list(box.xyxy[0].to('cpu').numpy().astype('int'))
            x_min, y_min, x_max, y_max = xyxy
            crop_xyxy.append([x_min, y_min, x_max, y_max])
            # crop된 이미지 기록
            crop_imgs.append(image[y_min:y_max, x_min:x_max,:].tolist())
            # crop된 이미지별 conf 기록
            crop_conf.append(box.conf.detach().cpu().numpy().astype('float32')[0])

        
        # crop된 object 이름 기록
        df_report['Crop_classes'].iloc[id] = crop_classes
        df_report['Crop_imgs'].iloc[id] = crop_imgs
        df_report['Crop_xyxy'].iloc[id] = crop_xyxy
        df_report['Crop_conf'].iloc[id] = crop_conf

        # detect을 통한 Category분류
        # banner가 frame안에 있으면 public
        # -1:초기화, 0:프레임, 1:공익, 2:정치 3:기타 
        # frame이면 0 아니면 -1(초기화)
        df_report['Category'].iloc[id] = [0 if class_name=='frame' else -1 for class_name in crop_classes]
        df_report['Category_basis'].iloc[id] = [0 if class_name=='frame' else -1 for class_name in crop_classes]

        df_report = check_category(df_report,id,image,crop_classes,crop_xyxy)

    return df_report

# 이미지를 종류별로 생성
def more_images(img):
    ocr = PaddleOCR(lang = 'korean')
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOT_BGR2GRAY)

    return rgb_img, gray_img

# 좌표, 글자, 정확도를 리스트 형태로 저장
def get_words(result):
    corrs = [temp[0] for temp in result[0]]  # 좌표들을 corrs에 저장
    texts = [temp[1][0] for temp in result[0]]  # 글자들을 texts에 저장
    scores = [temp[1][1] for temp in result[0]]  # 정확도를 scores에 저장

    return corrs, texts, scores


# 어떤 글자들을 인식해서 box를 쳤고, 그 box들이 쳐진 image와 세부 정보까지 쓰여진 image를 리턴하는 함수
def show_ocr(result, img_path, ocr):
    img = cv2.imread(img_path)
    corrs, texts, scores = get_words(result)
    result_simple = draw_ocr(image = img, boxes = corrs)
    result_details = draw_ocr(image = img, boxes = corrs, txts = texts, scores = scores, font_path = 'NanumSquareNeo-Variable.ttf')

    return result_simple, result_details

# 인식한 글자들을 y축을 기준으로 정렬한 후, x축을 기준으로 다시금 정렬해 상단-좌단 순으로 출력
def cluster_by_lines(corrs, texts):
    cor_texts = []
    for i in range(len(corrs)):
        cor_texts.append([corrs[i][0], texts[i]])

    cor_texts = sorted(cor_texts, key=lambda x:x[0][1])

    text_result = []

    try:
        current_group = [cor_texts[0]]

        for i in range(1, len(cor_texts)):
            if abs(cor_texts[i-1][0][1] - cor_texts[i][0][1]) < 15:
                current_group.append(cor_texts[i])
            else:
                text_result.append(current_group)
                current_group = [cor_texts[i]]
    except:
        current_group = []

    text_result.append(current_group)

    sorted_bundle = []
    for texts in text_result:
        texts = sorted(texts, key=lambda x:x[0][0])
        sorted_bundle.append(texts)

    contents = []
    for bundle in sorted_bundle:
        texts_only = []
        for words in bundle:
            texts_only.append(words[1])

        contents.append(' '.join(texts_only))

    return contents

# 이미지 처리 : contents만 출력
def get_contents(img_path):
    result = ocr.ocr(img = img_path, cls = False)

    corrs, texts, _ = get_words(result)
    contents = cluster_by_lines(corrs, texts)

    return contents

def Paddleocr_run(idx,df_report):
    # print("OCR 프로세스...")
    n_crops = len(df_report.iloc[idx]['Crop_classes'])
    
    for i in range(n_crops):
        # 만약 'banner'면 해당 사진의 contents를 추출해 저장
        if df_report.loc[idx]['Crop_classes'][i] == 'banner':
            contents = get_contents(np.array(df_report.loc[idx,'Crop_imgs'][i],dtype=np.uint8))
            df_report.loc[idx,'PaddleOCR_text'].append(contents)
        # frame 이면 빈 list 저장 (인덱스 맞춰주어야함)
        else:
            df_report.loc[idx,'PaddleOCR_text'].append(['frame'])

    return df_report

import uuid
import requests

api_url = ''
secret_key = ''

def clova_ocr(img_path):
    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
    ('file', open(img_path,'rb'))
    ]

    headers = {
    'X-OCR-SECRET': secret_key
    }

    response = requests.request("POST", api_url, headers=headers, data = payload, files = files)

    return response

def get_clova_contents(img_path):
    response = clova_ocr(img_path)
    response = response.json()
    contents = []
    for field in response['images'][0]['fields']:
        text = field['inferText']
        contents.append(text)

    return contents

def clova_ocr_run(idx, df_report):
    # print("OCR 프로세스...")
    n_crops = len(df_report.iloc[idx]['Crop_classes'])
    
    for i in range(n_crops):
        # 만약 'banner'면 해당 사진의 contents를 추출해 저장
        if df_report.loc[idx]['Crop_classes'][i] == 'banner':
            img_path = 'naver_ocr_temp.jpg'
            image = Image.fromarray(np.uint8(df_report.loc[idx]['Crop_imgs'][i]))
            image.save(img_path, 'jpeg')
            contents = get_clova_contents(img_path)
            df_report.loc[idx,'ClovaOCR_text'].append(contents)
        # frame 이면 빈 list 저장 (인덱스 맞춰주어야함)
        else:
            df_report.loc[idx,'ClovaOCR_text'].append(['frame'])

    return df_report

# chatGPT 키 입력 후 실행할 때
import openai

openai.api_key = ""

# -1:초기화, 0:프레임, 1:공익, 2:정치 3:기타 
def classify_text(text):
    text = ' '.join(text)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are responsible for classifying the text of advertising banners near the road or on the street."},

            {"role": "system", "content": "There are a total of three classes of advertising banners to classify."},

            {"role": "system", "content": "The Class 1 is the text of the public service banner installed by the city hall and district office."},
            {"role": "system", "content": "The Class 2 is  the text of a political promotion banner set up by politicians."},
            {"role": "system", "content": "The Class 3 is all banners other than 1 and 2. For example, text such as a hospital, gym, or academy promotional banner."},

            {"role": "system", "content": "The text I deliver is a set of words in the form of a list, and please combine and guess the words to classify the class."},

            {"role": "user", "content": f"The text I want to convey is: {text}."},
            {"role": "assistant", "content": f"Please provide a classification: 1, 2, or 3 based on the content you just shared."}
        ]
        
    )
    return response.choices[0].message['content']
def chatGPT_run(idx,df_report):
    print("chatGPT 프로세스...")

    n = len(df_report.loc[idx]['Crop_imgs'])
    crop_class_names = df_report.loc[idx]['Crop_classes']
    crop_texts = df_report.loc[idx]['ClovaOCR_text'].copy()

    # -1:초기화, 0:프레임, 1:공익, 2:정치 3:기타 
    categories = df_report.loc[idx]['Category']
    categories_basis = df_report.loc[idx]['Category_basis']
    for i in range(n):
        print("클래스명 : ", crop_class_names[i])
        if categories[i] == 0:
            print("내용 : *frame 입니다. (Detect 기반)")
            print(f"내용 : {crop_texts[i]} (카테고리: {0})")
            continue
        elif categories[i] == 1:
            print("내용 : *pulbic 입니다. (Detect 기반)")
            print(f"내용 : {crop_texts[i]} (카테고리: {1})")
        else:
            category_text = classify_text(crop_texts[i])  # OCR 텍스트를 GPT-3로 분류합니다.
            if 'Class 1' in category_text:
                categories[i] = 1
            elif 'Class 2' in category_text:
                categories[i] = 2
            else:
                categories[i] = 3
            categories_basis[i] = category_text
            
    df_report.loc[idx]['Category'] = categories
    df_report.loc[idx]['Category_basis'] = categories_basis

    return df_report

# imgs = 이미지 데이터 경로

imgs = glob('capture_data/*.jpg')

for idx,img in enumerate(imgs):

    # object detection
    df_report = yolo_run(img,df_report)

    # ocr
    df_report = Paddleocr_run(idx,df_report)
    df_report = clova_ocr_run(idx,df_report)
    
    # text classification
    df_report = chatGPT_run(idx,df_report)

df_report.to_csv('reports/report_'+'_'.join(self.date_created)+'.csv')