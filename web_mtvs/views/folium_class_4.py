
import pandas as pd
import base64
import folium
import folium.plugins

import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import branca

class MapManager:
    # 임시 좌표를 판교역 근처 좌표로 수정
    def __init__(self,df_report):
        self.df_report = df_report
        self.locations = ['[37.3953946671528, 127.1093297131001]', '[37.39417823717656, 127.1093844115068]', '[37.39271861487178, 127.10934841208203]', '[37.39185066864326, 127.11250910446297]', '[37.39345442116072, 127.11257925899302]', '[37.395544745716585, 127.11262755830957]', '[37.39650119524654, 127.111183435904]', '[37.39649911490269, 127.11337434990375]', '[37.39601079572209, 127.11520313115568]', '[37.39440567382231, 127.11653325709659]', '[37.39346684071838, 127.11831608687358]', '[37.39080112282156, 127.11700196362115]', '[37.39188034187261, 127.119013740367]', '[37.3940362949143, 127.10695620640082]', '[37.396431672499325, 127.10842774919402]', '[37.397943736734206, 127.11021431692127]']
       
        # self.df_report['Location'] = locations
        # self.map = folium.Map(location = [37.566697, 126.978426], control_scale=True, zoom_start=12)


    # origin_img를 저장하고, 경로 추출
    def load_origin_img_path(self, df_report, idx):
        origin_img = (df_report.loc[idx,'Origin_img'])[0]
        origin_img = np.array(origin_img,dtype=np.uint8)
        img = Image.fromarray(origin_img)
        img_path = 'folium/origin_img.jpg'
        img.save(img_path, 'jpeg')

        return img_path

        # detect_img를 저장하고, 경로 추출
    def load_detect_img_path(self, df_report, idx):
        detect_img = (df_report.loc[idx,'Detect_img'])
        detect_img = np.array(detect_img,dtype=np.uint8)
        img = Image.fromarray(detect_img)
        img_path = 'folium/detect_img.jpg'
        img.save(img_path, 'jpeg')

        return img_path

        # crop된 이미지들을 저장하고, 경로 추출
    def load_crop_imgs_path(self,df_report, idx):
        crop_imgs = (df_report.loc[idx]['Crop_imgs'])
        n_crops = len(crop_imgs)
        crop_path_list = []

        for i in range(n_crops):
            crop_img = np.array(crop_imgs[i],dtype=np.uint8)
            img = Image.fromarray(crop_img)
            img_path = 'folium/crop_images_' + str(i) + '.jpg'
            img.save(img_path, 'jpeg')
            crop_path_list.append(img_path)

        return crop_path_list

        # 모든 이미지 경로를 저장하고, 경로 추출
    def load_imgs_path(self, df_report, idx):
        base_path_list = []
        crop_path_list = []

        origin_img_path = self.load_origin_img_path(df_report, idx)
        detect_img_path = self.load_detect_img_path(df_report, idx)
        base_path_list.append(origin_img_path)
        base_path_list.append(detect_img_path)

        try:
            crop_path_list = self.load_crop_imgs_path(df_report,idx)
        except:
            pass

        return base_path_list, crop_path_list

        # 저장된 이미지를 경로를 통해 불러온 후, html에 넣기 위해 base64 형식으로 인코딩한 다음 디코딩
    # 기본 이미지와 detect 이미지는 반드시 존재하지만, crop 이미지는 경우에 따라 존재하지 않을 수도 있어 함수를 나눔
    def get_base_pics(self, base_path_list):
        pic_base = []
        for img_path in base_path_list:
            pic = base64.b64encode(open(img_path, 'rb').read()).decode()
            pic_base.append(pic)

        return pic_base

    # 저장된 이미지를 경로를 통해 불러온 후, html에 넣기 위해 base64 형식으로 인코딩한 다음 디코딩
    def get_crop_pics(self, crop_path_list):
        pic_crops = []
        try:
            for img_path in crop_path_list:
                pic = base64.b64encode(open(img_path, 'rb').read()).decode()
                pic_crops.append(pic)
        except:
            pass

        return pic_crops

        # crop된 이미지들을 html 표 형식으로 시각화하는 함수
    def get_crop_htmls(self, df_report, idx, crop_path_list, pic_crops):
        htmls = []
        for i in range(len(crop_path_list)):
            if (df_report.loc[idx]['Crop_classes'])[i] == 'frame':
                category = (df_report.loc[idx]['Category'])[i]
                categories = {-1:'초기화', 0:'프레임', 1:'합법(공익)', 2:'정치', 3:'기타'}
                texts = (df_report.loc[idx]['ClovaOCR_text'])[i]
                basis = (df_report.loc[idx]['Category_basis'])[i]

                # 이미지 옆에 표 형태로 세부내용 넣기
                html = f"""
                <tr>
                    <tr>
                        <td>
                        <table id="image">
                            <tr>
                                <td><img src="data:image/jpeg;base64,{pic_crops[i]}" width=150 height=75></td>
                            </tr>
                            <tr>
                                <td align=center border=1><font size=1>frame</font></td>
                            </tr>
                        </table>
                        </td>
                        <td>    
                        <table border=1 width=425 height=95%>
                            <tr><td align=center width=40><font size=1>범례</font></td><td><font size=1>{categories[category]}</font></td></tr>
                            <tr><td align=center width=40<font size=1>내용</font></td><td class=visible><font size=1>{texts}</font></td></tr>
                            <tr><td align=center width=40><font size=1>근거</font></td><td class=visible><font size=1>{basis}</font></td></tr>
                        </table>
                        </td>                    
                    </tr>
                </tr>
                """
                htmls.append(html)
            else:
                category = (df_report.loc[idx]['Category'])[i]
                categories = {-1:'초기화', 0:'프레임', 1:'합법(공익)', 2:'정치', 3:'기타'}
                texts = (df_report.loc[idx]['ClovaOCR_text'])[i]
                basis = (df_report.loc[idx]['Category_basis'])[i]

                # 이미지 옆에 표 형태로 세부내용 넣기
                html = f"""
                <tr>
                    <tr>
                        <td>
                        <table id="image">
                            <tr>
                                <td><img src="data:image/jpeg;base64,{pic_crops[i]}" width=150 height=75></td>
                            </tr>
                            <tr>
                                <td align=center border=1><font size=1>crop_img_{i}</font></td>
                            </tr>
                        </table>
                        </td>
                        <td>    
                        <table border=1 width=425 height=95%>
                            <tr><td align=center width=40><font size=1>범례</font></td><td><font size=1>{categories[category]}</font></td></tr>
                            <tr><td align=center width=40><font size=1>내용</font></td><td class=visible><font size=1>{texts}</font></td></tr>
                            <tr><td align=center width=40><font size=1>근거</font></td><td class=visible><font size=1>{basis}</font></td></tr>
                        </table>
                        </td>                    
                    </tr>
                </tr>
                """
                htmls.append(html)

        return htmls

        # popup에 html표를 집어넣기
    def get_folium_popups(self, df_report, idx):
        base_path_list, crop_path_list = self.load_imgs_path(df_report, idx)
        pic_base = self.get_base_pics(base_path_list)
        pic_crops = self.get_crop_pics(crop_path_list)

        base_h = len(pic_base) * 110
        crop_h = len((df_report.loc[idx]['Crop_classes'])) * 110  # 만약 banner나 frame을 detect하지 못했다면 popup창의 크기를 조절

        crop_html = f''
        try:
            htmls = self.get_crop_htmls(df_report, idx, crop_path_list, pic_crops)
            if htmls != []:
                for i in range(len(htmls)):
                    crop_html += htmls[i] + '\n'
        except:
            pass

        # 원본 이미지와 detect 이미지는 고정적으로 들어감
        image_tag = f"""
        <head>
        <meta charset="euc-kr">
        <link rel="stylesheet" href="">
        <style>
            <link rel="stylesheet" href="static/css/style.css">
        </style>
        </head>
        <body>
            <div>
            <table class="table-dark" border=1>
                <tr align=center>
                    <th width="150">촬영 일시</th>
                    <td width="425">ID 번호: {df_report.loc[idx]['ID']} / {df_report.loc[idx]['Date']} {df_report.loc[idx]['Time']}</td>
                </tr>
                <tr align=center>
                    <th width="150">이미지</th>
                    <th width="425">내용</th>
                </tr>
                <tr>
                    <td>
                    <table id="image">
                        <tr>
                            <td><img src="data:image/jpeg;base64,{pic_base[0]}" width=150 height=75></td>
                        </tr>
                        <tr>
                            <td align=center><font size=1>Origin Image</font></td>
                        </tr>
                    </table>
                    </td><td align=center><font size=1>원본 이미지</font></td>
                </tr>
                <tr>
                    <td>
                    <table id="image">
                        <tr>
                            <td><img src="data:image/jpeg;base64,{pic_base[1]}" width=150 height=75></td>
                        </tr>
                        <tr>
                            <td align=center><font size=1>Detect Image</font></td>
                        </tr>
                    </table>
                    </td><td align=center><font size=1>Yolo로 Detect한 이미지</font></td>
                </tr>
        """ + crop_html + """
        </body>
        """

        height = 59 + base_h + crop_h
        if height > 450:
            height = 450

        iframe = branca.element.IFrame(image_tag, width=627, height=height)
        popup = folium.Popup(iframe)

        return popup


    # 원본 이미지, detect 이미지, crop 이미지들을 저장할 폴더 생성
    def create_folder(self, df_report):
        dir_path = 'folium/'
        os.makedirs(dir_path, exist_ok=True)
        # df_report['Location'] = self.locations
        print("lie")
        # map = folium.Map(location = [37.566697, 126.978426], control_scale=True, zoom_start=12)

        center = (df_report.iloc[0]['Location'])  # 지도의 중심좌표
        map = folium.Map(location = center, control_scale=True, zoom_start=12)

        # 각 마커별 그룹 생성
        political_group = folium.FeatureGroup(name="정치").add_to(map)
        illegal_group = folium.FeatureGroup(name="불법").add_to(map)
        legal_group = folium.FeatureGroup(name="합법").add_to(map)
        print("GROUP")
        # 마커별 on/off가 가능하게 해주는 코드
        folium.LayerControl(collapsed=False).add_to(map)
        print("on")

        for i in tqdm(range(len(df_report))):
            popup = self.get_folium_popups(df_report, i)

            # category에 따라 마커별 그룹에 더하기
            # 2: 정치, 3: 불법, 그외: 합법
            if 2 in (df_report.iloc[i]['Category']):
                political_group.add_child(folium.Marker(location = (df_report.iloc[i]['Location']), icon = folium.Icon(color = 'blue', icon = 'bookmark'), popup = popup))
                print("Mark2")
            elif 3 in (df_report.iloc[i]['Category']):
                illegal_group.add_child(folium.Marker(location = (df_report.iloc[i]['Location']), icon = folium.Icon(color = 'red', icon = 'star'), popup = popup))
                print("Mark3")
            else:
                legal_group.add_child(folium.Marker(location = (df_report.iloc[i]['Location']), icon = folium.Icon(color = 'green', icon = 'check'), popup = popup))
                print("Mark1")

        map.save('./web_mtvs/templates/Map.html')