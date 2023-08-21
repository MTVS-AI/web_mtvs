import base64
import pandas as pd
import folium
import numpy as np
from PIL import Image

class MapManager:
    def __init__(self, report_file_path):
        self.df = pd.read_csv(report_file_path)
# df = pd.read_csv('reports/report_2023_08_12.csv')

    # origin_img를 저장하고, 경로 추출
    def load_origin_img_path(self, idx):
        origin_img = eval(self.df.loc[idx, 'Origin_img'])[0]
        origin_img = np.array(origin_img,dtype=np.uint8)
        img = Image.fromarray(origin_img)
        img_path = 'origin_img.jpg'
        img.save(img_path, 'jpeg')

        return img_path

    # detect_img를 저장하고, 경로 추출
    def load_detect_img_path(self, idx):
        detect_img = eval(self.df.loc[idx, 'Detect_img'])
        detect_img = np.array(detect_img,dtype=np.uint8)
        img = Image.fromarray(detect_img)
        img_path = 'detect_img.jpg'
        img.save(img_path, 'jpeg')

        return img_path

    # crop된 이미지들을 저장하고, 경로 추출
    def load_crop_imgs_path(self,idx):
        crop_imgs = eval(self.df.loc[idx]['Crop_imgs'])
        n_crops = len(crop_imgs)
        crop_path_list = []

        for i in range(n_crops):
            crop_img = np.array(crop_imgs[i],dtype=np.uint8)
            img = Image.fromarray(crop_img)
            img_path = 'crop_images_' + str(i) + '.jpg'
            img.save(img_path, 'jpeg')
            crop_path_list.append(img_path)

        return crop_path_list

    # 모든 이미지 경로를 저장하고, 경로 추출
    def load_imgs_path(self, idx):
        base_path_list = []
        crop_path_list = []

        origin_img_path = self.load_origin_img_path(idx)
        detect_img_path = self.load_detect_img_path(idx)
        base_path_list.append(origin_img_path)
        base_path_list.append(detect_img_path)

        try:
            crop_path_list = self.load_crop_imgs_path(idx)
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
    def get_crop_htmls(self, idx, crop_path_list, pic_crops):
        htmls = []
        for i in range(len(crop_path_list)):
            if eval(self.df.loc[idx]['Crop_classes'])[i] == 'frame':
                category = eval(self.df.loc[idx]['Category'])[i]
                categories = {-1:'초기화', 0:'프레임', 1:'합법(공익)', 2:'정치', 3:'기타'}
                texts = eval(self.df.loc[idx]['ClovaOCR_text'])[i]
                basis = eval(self.df.loc[idx]['Category_basis'])[i]

                # 이미지 옆에 표 형태로 세부내용 넣기
                html = f"""
                <tr>
                    <td><img src="data:image/jpeg;base64,{pic_crops[i]}" width=200 height=100></td><td align=center>frame</td>
                    <td>
                    <table border width=650 height=95%>
                        <tr><td align=center width=40>범례</td><td>{categories[category]}</td></tr>
                        <tr><td align=center width=40>내용</td><td class=visible>{texts}</td></tr>
                        <tr><td align=center width=40>근거</td><td class=visible>{basis}</td></tr>
                    </table>
                    </td>
                </tr>
                """
                htmls.append(html)
            else:
                category = eval(self.df.loc[idx]['Category'])[i]
                categories = {-1:'초기화', 0:'프레임', 1:'합법(공익)', 2:'정치', 3:'기타'}
                texts = eval(self.df.loc[idx]['ClovaOCR_text'])[i]
                basis = eval(self.df.loc[idx]['Category_basis'])[i]

                # 이미지 옆에 표 형태로 세부내용 넣기
                html = f"""
                <tr>
                    <td>
                    <img src="data:image/jpeg;base64,{pic_crops[i]}" width=200 height=100></td><td align=center>crop_img_{i}</td>
                    <td>
                    <table border width=650 height=95%>
                        <tr><td align=center width=40>범례</td><td>{categories[category]}</td></tr>
                        <tr><td align=center width=40>내용</td><td class=visible>{texts}</td></tr>
                        <tr><td align=center width=40>근거</td><td class=visible>{basis}</td></tr>
                    </table>
                    </td>
                </tr>
                """
                htmls.append(html)

        return htmls

    # popup에 html표를 집어넣기
    def get_folium_popups(self, idx):
        base_path_list, crop_path_list = self.load_imgs_path(idx)
        pic_base = self.get_base_pics(base_path_list)
        pic_crops = self.get_crop_pics(crop_path_list)

        base_h = len(pic_base) * 100
        crop_h = len(eval(self.df.loc[idx]['Crop_classes'])) * 100  # 만약 banner나 frame을 detect하지 못했다면 popup창의 크기를 조절

        crop_html = f''
        try:
            htmls = self.get_crop_htmls(self.df, idx, crop_path_list, pic_crops)
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
        <style src="">
        </style>
        </head>
        <body>
            <div>
            <table border=2>
                <tr align=center>
                    <th>이미지</th>
                    <th>종류</th>
                    <th>내용</th>
                </tr>
                <tr>
                    <td><img src="data:image/jpeg;base64,{pic_base[0]}" width=200 height=100></td><td align=center width=75>Origin Image</td><td align=center>원본 이미지</td>
                </tr>
                <tr>
                    <td><img src="data:image/jpeg;base64,{pic_base[1]}" width=200 height=100></td><td align=center width=75>Detect Image</td><td align=center>YOLO로 Detect한 이미지</td>
                </tr>
        """ + crop_html + """
        </body>
        """

        iframe = folium.IFrame(image_tag, width=1000, height=150+base_h+crop_h)
        popup = folium.Popup(iframe)

        return popup