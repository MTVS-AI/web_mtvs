from flask import Flask
from .views import main_views, myhome_views, maps_views

def create_app():
    app = Flask(__name__)
    # app.secret_key = "secretdlsco123"
    
    # Blueprints 등록
    app.register_blueprint(main_views.bp)
    app.register_blueprint(myhome_views.bp)
    app.register_blueprint(maps_views.bp)

    return app
