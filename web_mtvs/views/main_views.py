from flask import Blueprint, render_template#, url_for
# from werkzeug.utils import redirect
# from pybo.models import Question
# blueprint : 주소 관리 역할

bp = Blueprint('main', __name__, url_prefix='/') 

@bp.route('/')
def index():
    return render_template('index.html')