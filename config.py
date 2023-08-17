import os

BASE_DIR = os.path.dirname(__file__)

SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR, 'pybo.db'))
# 데이터베이스에 접근 가능한 주소 설정, sqlalchemy : db 관리 역할
# 해당 경로에 pybo.db database 생성

# sqlalchemy event 설정
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY ="dev"