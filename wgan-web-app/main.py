import os
import cv2
import time
import json
import yaml
import torch
import random
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, jsonify
import subprocess
import mysql.connector
from utils.data_loader import get_data_loader
from utils.config import parse_args
from models.gan import GAN
from models.dcgan import DCGAN_MODEL
from models.wgan_clipping import WGAN_CP
from models.wgan_gradient_penalty import WGAN_GP
import traceback
# from mobilenetmultilabelclassification.model import MultiOutputModel
from config import DtatBaseConf  # getAttributesDataset, attributes_file,
# from mobilenetmultilabelclassification.inference import main

app = Flask(__name__)
app.config['UPLOAD_IMAGE'] = r'./images'  #上传图像路径
app.config['RESULT_SAVE_IMAGE'] = r'./images'  #结果图像路径

# 在app.config中设置好连接数据库的信息；然后在SQLAlchemy(app)创建一个db对象；SQLAlchemy会自动的读取app.config中连接数据库的信息
DB_URI = ""
labels_list = ["color", "gender", "article"]
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(DtatBaseConf["USERNAME"],
                                                              DtatBaseConf["PASSWORD"],
                                                              DtatBaseConf["HOSTNAME"],
                                                              DtatBaseConf["PORT"],
                                                              DtatBaseConf["DATABASE"])
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
db = SQLAlchemy(app)

class AIReco(db.Model):
    __tablename__ = "AIAgriculture"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    picname = db.Column(db.String(100), unique=True, nullable=False)
    result = db.Column(db.String(100), nullable=False)
    requesttime = db.Column(db.String(100), nullable=False)
    costTime = db.Column(db.String(100), nullable=False)


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def avi_to_web_mp4(input_file_path):
    '''
    ffmpeg -i test_result.avi -vcodec h264 test_result.mp4
    @param: [in] input_file_path 带avi或mp4的非H264编码的视频的全路径
    @return: [output] output_file_path 生成的H264编码视频的全路径
    '''
    output_file_path = input_file_path[:-3] + 'mp4'
    cmd = 'ffmpeg -y -i {} -vcodec h264 {}'.format(input_file_path, output_file_path)
    subprocess.call(cmd, shell=True)
    return output_file_path


def clear_pic(file_path):
    """
    清空图像
    :param file_path:
    :return:
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def clear_input_pic(file_path=None):
    """
    清空检测图像
    :return:
    """
    # 如果存在上传文件，则删除
    clear_pic(file_path)


def plot_one_box(img, color=None, label=None, line_thickness=None):
    """
    将检测到的结果绘制在源图像中
    :param x:
    :param img:
    :param color:
    :param label:
    :param line_thickness:
    :return:
    """
    if img is not None:
        if isinstance(label, dict):
            label_cnt = 0
            for key in label.keys():
                color = color or tuple((random.randint(0, 255) for _ in range(3)))
                if label:
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img)
                    fontText = ImageFont.truetype('./static/fonts/STKAITI.TTF', 10, encoding="utf-8")
                    draw.text((10, 10 + label_cnt * 15), "{}: {}".format(key, label[key]),  color, font=fontText)
                    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                    label_cnt += 1
            return img
        else:
            color = color or tuple((random.randint(0, 255) for _ in range(3)))
            if label:
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img)
                fontText = ImageFont.truetype('./static/fonts/STKAITI.TTF', 100, encoding="utf-8")
                draw.text((120, 120),  label, color, font=fontText)
                img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            return img


def det_result_vis(result, image):
    """
    结果可视化
    :param result:
    :param image_filepath:
    :return:
    """
    if result["name"]:
        image = plot_one_box(image,
                             color=None,
                             label=result["name"],
                             line_thickness=None)
        return image
    else:
        return None


@app.route('/upload', methods=['post'])
def upload_image():
    """
    上传图像
    :return:
    """
    img = request.files.get('upfile')  # 获取文件内容
    path = app.config['UPLOAD_IMAGE'] + img.filename
    # clear_input_pic(path)  # 如果存在文件，则删除
    img.save(path)  # 保存文件
    return ''


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generic', methods=['GET','POST'])
def generic():
    image_url = image_url = ("./static/images/dgan_model_image.png")
    return render_template('generic.html',image_url=image_url)

@app.route('/elements', methods=['GET','POST'])
def elements():
    return render_template('elements.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    data = request.get_json()
    model_type = data.get('model', None)
    is_train = data.get('is_train', False)
    download = data.get('download', False)
    dataroot = data.get('dataroot', 'default_dataroot')
    dataset = data.get('dataset', 'default_dataset')
    generator_iters = data.get('generator_iters', 10000)
    cuda = data.get('cuda', False)
    batch_size = data.get('batch_size', 64)
    load_D = data.get('load_D', None)[:-4]+model_type+'.pkl'
    load_G = data.get('load_G', None)[:-4]+model_type+'.pkl'
    if model_type == 'GAN':
        model = GAN(is_train=is_train, download=download, dataroot=dataroot,
                        dataset=dataset, generator_iters=generator_iters,
                        cuda=cuda, batch_size=batch_size, load_D=load_D, load_G=load_G)
    elif model_type == 'DCGAN':
        model = DCGAN_MODEL(is_train=is_train, download=download, dataroot=dataroot,
                        dataset=dataset, generator_iters=generator_iters,
                        cuda=cuda, batch_size=batch_size, load_D=load_D, load_G=load_G)
    elif model_type == 'WGAN-CP':
        model = WGAN_CP(is_train=is_train, download=download, dataroot=dataroot,
                        dataset=dataset, generator_iters=generator_iters,
                        cuda=cuda, batch_size=batch_size, load_D=load_D, load_G=load_G)
    elif model_type == 'WGAN-GP':
        model = WGAN_GP(is_train=is_train, download=download, dataroot=dataroot,
                        dataset=dataset, generator_iters=generator_iters,
                        cuda=cuda, batch_size=batch_size, load_D=load_D, load_G=load_G)
    else:
        return jsonify({'error': 'Model type non-existing. Try again.'}), 400
    test_loader = get_data_loader(dataset,batch_size,dataroot,download)
    model.evaluater(test_loader, load_D, load_G)
    img = Image.open('./static/images/dgan_model_image.png')
    enlarged_img = img.resize((256, 256), Image.BICUBIC)
    enlarged_img.save('./static/images/1.png')
    image_url=image_url=("./static/images/dgan_model_image.png")
    return render_template('generic.html',image_url=image_url)

@app.route('/login', methods=['GET','POST'])
def all():
    cnx = mysql.connector.connect(user=os.environ.get('MYSQL_USER','root'), password=os.environ.get('MYSQL_PASSWORD',''), host=os.environ.get('MYSQL_HOST','localhost'), database='c4')
    cursor = cnx.cursor()
    username = request.form['username']
    password = request.form['password']
    query = "SELECT * FROM user WHERE name = %s AND password = %s"
    cursor.execute(query, (username, password))
    result = cursor.fetchone()
    if result:
        return render_template('index.html')
    else:
        return render_template('indexall.html')

@app.route('/register', methods=['GET','POST'])
def register():
    cnx = mysql.connector.connect(user=os.environ.get('MYSQL_USER','root'), password=os.environ.get('MYSQL_PASSWORD',''), host=os.environ.get('MYSQL_HOST','localhost'), database='c4')
    cursor = cnx.cursor()
    username = request.form['username']
    password = request.form['password']
    query = "INSERT INTO user (name, password) VALUES (%s, %s)"
    cursor.execute(query, (username, password))
    cnx.commit()
    cursor.close()
    cnx.close()
    return render_template('indexall.html')

if __name__ == "__main__":
    app.run(port=5000, debug=True)
