import tensorflow as tf

from flask import Blueprint, render_template, request

from src.init import init
from src.translate import translate

bp = Blueprint('', __name__, url_prefix='/')


@bp.route('/')
def index(kor=None):
    if request.method == 'POST':
        return render_template('predict.html', kor=kor)
    elif request.method == 'GET':
        temp = request.args.get('kor')

        if temp != None:
            init_dic = init()
            # checkpoint_dir내에 있는 최근 체크포인트(checkpoint)를 복원합니다.
            init_dic['checkpoint'].restore(tf.train.latest_checkpoint(init_dic['checkpoint_dir']))
            result = translate(temp, init_dic)
            return render_template('predict.html', kor=result)

        return render_template('predict.html', kor=kor)