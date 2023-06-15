# main.py
import shutil

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import carbon_classifier  # Import your inference function

app = Flask(__name__)

# model_inference.py

import os
import argparse
import random
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import torchvision
import torchvision.models as models
from torchvision.models import mobilenet_v3_large
import torchvision.transforms as transforms

import sklearn
from sklearn import metrics, preprocessing
from sklearn.metrics import f1_score, confusion_matrix

"""랜덤 수의 Seed 고정, 재현성 보장"""


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


""" Pre-trained 모델 호출 및 선언"""


class Model(nn.Module):
    def __init__(self, model_arch_str, num_classes=2, pretrained=True):
        super(Model, self).__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)  ## 모델 선언 여기 models.##(pretrained =pretrained)
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x


""" 1, 2차 모델 선언 """


def model_define(category):
    """ 1차 분류 ('10Kwalk', 'battery', 'box', 'else', 'bottle', 'handkerchief',
 'milk', 'paper', 'pet', 'plug', 'receipt', 'shopping bag', 'stairs',
 'transportation', 'trash picking', 'dishes' """
    class_num = 16
    model_path = 'mobilenet_v3_16classes.pth'

    if category == 'bottle':
        """ bottle (양치컵 사용하기, 텀블러 사용하기) """
        class_num = 2
        model_path = 'mobilenet_v3_bottle.pth'

    elif category == 'dishes':
        """ dishes (랩 쓰지 않기, 잔반 남기지 않기, 채소식단) """
        class_num = 3
        model_path = 'mobilenet_v3_dishes.pth'

    elif category == 'box':
        """ 테이브 제거 검증 """
        class_num = 2
        model_path = 'mobilenet_v3_box.pth'

    elif category == 'pet':
        """ 페트병 라벨 제거 검증 """
        class_num = 2
        model_path = 'mobilenet_v3_pet.pth'
    model = Model('mobilenet_v3_large', class_num, pretrained=True)
    model.load_state_dict(torch.load('../models/' + model_path, map_location=torch.device('cpu')))

    return model


""" 이미지 분석을 위한 기본 전처리 """
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


def load_img(image_path):
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_image)
    return input_tensor.unsqueeze(0)


def decode_category(category, result):
    if category == 'bottle':
        class_labels = ["toothcup", "tumbler"]
    elif category == 'pet':
        class_labels = ["unlabeled", "labeled"]
    elif category == 'box':
        class_labels = ["box", "untapedBox"]
    elif category == 'dishes':
        class_labels = ['wrap', 'leftover', 'green dish']
    else:
        print('here')
        class_labels = ['10Kwalk', 'battery', 'box', 'else', 'bottle', 'handkerchief',
                        'milk', 'paper', 'pet', 'plug', 'receipt', 'shopping bag', 'stairs',
                        'transportation', 'trash picking', 'dishes']

    le = preprocessing.LabelEncoder()
    le.fit_transform(class_labels)

    return le.inverse_transform([result])


def infer_from_model(image_path):
    """ 추론  """

    def inference(images):
        """ Move the images to the specified device (GPU or CPU) """
        images = images.to(device)

        """ Disable gradient calculation and enable inference mode """
        with torch.no_grad():
            """ Perform the forward pass """
            outputs = model(images)
        """ Apply softmax to obtain class probabilities """
        probabilities = torch.softmax(outputs, dim=1)

        return probabilities

    image_path = image_path

    seed_everything(42)
    category = ''
    # image_path = './tst_img/toothcup.jpg' # Load your input images here

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_define(category)
    model.eval()
    model = model.to(device)

    input_batch = load_img(image_path)
    input_batch = input_batch.to(device)

    # Run inference
    output_probabilities = inference(input_batch)
    predicted_class_index = torch.argmax(output_probabilities, dim=1)
    result = predicted_class_index.item()
    result = decode_category(category, result)

    print(f'first classification: {result}')
    category = result
    if category in ['bottle', 'pet', 'box', 'dishes']:
        model = model_define(category)
        model = model.to(device)
        output_probabilities = inference(input_batch)
        predicted_class_index = torch.argmax(output_probabilities, dim=1)
        result = predicted_class_index.item()
        result = decode_category(category, result)
        print(f'final classification: {result}')

    # print(f'result : {result}')
    return result

def final_result(prediction):
    if prediction == 'labeled':
        return '라벨이 제거 되지 않은 페트병(labeled pet)'
    elif prediction == 'unlabeled':
        return '라벨이 제거된 페트병(unlabeled pet)'
    elif prediction == 'toothcup':
        return '양치컵 사용(use tooth brush cup)'
    elif prediction == 'tumbler':
        return '일회용컵 대신 텀블러 사용(use tumbler)'
    elif prediction == 'untapedBox':
        return '테이프 제거된 박스(untaped box)'
    elif prediction =='box':
        return '테이프가 제거되지 않은 박스(taped box)'
    elif prediction == 'wrap':
        return '랩대신 뚜껑 사용(not using wrap)'
    elif prediction == 'leftover':
        return '잔반 남기지 않기(no leftover)'
    elif prediction == 'green dish':
        return '채소 식단(green dish)'
    elif prediction == '10Kwalk':
        return '만보 걷기(walking 10km)'
    elif prediction == 'battery':
        return '휴대폰 절전모드 사용(low power mode)'
    elif prediction == 'handkerchief':
        return '손수건 사용(use handkerchief)'
    elif prediction == 'milk':
        return '우유팩 재활용(recycle milk-pack)'
    elif prediction == 'paper':
        return '이면지 사용(reuse paper)'
    elif prediction == 'plug':
        return '플러그 제거(unplug)'
    elif prediction == 'receipt':
        return '전자 영수증 받기(electric receipt'
    elif prediction == 'shopping bag':
        return '장바구니 사용(reusable shopping bag)'
    elif prediction == 'stairs':
        return '계단 이용(use stairs)'
    elif prediction == 'transportation':
        return '대중교통 이용(use public vehicle)'
    elif prediction == 'trash picking':
        return '쓰레기 줍기(trach picking)'
    else:
        return '분류 되지 않는 사진 (else)'


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_filename = secure_filename(image_file.filename)
            image_location = os.path.join('uploads', image_filename)
            image_file.save(image_location)
            prediction = infer_from_model(image_location)
            prediction = final_result(prediction)
            image_loc = '/static/' + image_filename  # 이미지 파일의 URL 경로

            # 이미지 파일을 static 폴더로 이동시킴
            static_image_location = os.path.join('static', image_filename)
            shutil.move(image_location, static_image_location)

            return render_template('main.html', prediction=prediction, image_loc=image_loc)
    return render_template('main.html', prediction=None, image_loc=None)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)