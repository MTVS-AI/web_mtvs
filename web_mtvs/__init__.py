from flask import Flask
from .views import main_views, myhome_views

def create_app():
    app = Flask(__name__)
    
    # Blueprints 등록
    app.register_blueprint(main_views.bp)
    app.register_blueprint(myhome_views.bp)

    return app
