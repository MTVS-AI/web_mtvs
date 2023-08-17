from flask import Flask
# from flask_migrate import Migrate
# from flask_sqlalchemy import SQLAlchemy
# import config

# db = SQLAlchemy()
# migrate = Migrate()

def create_app():
    app = Flask(__name__)
    # app.config.from_object(config) # config의 설정 값들이 flask app에 등록됨

    # db.init_app(app) # app 등록
    # migrate.init_app(app, db) # migrate : app과 db를 연결하는 역할
    
    # from . import models
    from .views import main_views, myhome_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(myhome_views.bp)

    return app