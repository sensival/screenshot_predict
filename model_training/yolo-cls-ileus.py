from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ultralytics import YOLO
import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns



# 데이터셋 경로 설정
dataset_path = 'C:/Users/wogns/OneDrive/바탕 화면/dl/ileus/train'
ileus_path = os.path.join(dataset_path, 'ILEUS')
normal_path = os.path.join(dataset_path, 'NORMAL')
output_directory = "C:/Users/wogns/OneDrive/바탕 화면/dl/output"
now_dir = os.path.abspath(os.path.dirname(__file__))

# 이미지 파일 리스트 생성
ileus_images = [os.path.join(ileus_path, f) for f in os.listdir(ileus_path) if f.endswith('.jpg')]
normal_images = [os.path.join(normal_path, f) for f in os.listdir(normal_path) if f.endswith('.jpg')]

# 전체 데이터 리스트와 레이블 리스트 생성
data = ileus_images + normal_images
labels = [0] * len(ileus_images) + [1] * len(normal_images)  # ILEUS = 0, NORMAL = 1

# 데이터와 레이블을 섞기
combined = list(zip(data, labels))
np.random.shuffle(combined)
data, labels = zip(*combined)

# k-fold 설정
k = 10
kf = KFold(n_splits=k)


model_path = 'yolov8n-cls.pt'  # 미리 학습된 YOLOv8n-cls 모델 경로

# YOLO 모델 로드
model = YOLO(model_path)

fold_idx = 0
for train_index, test_index in kf.split(data):
    # 각 폴드별로 데이터 분할
    train_data, test_data = [data[i] for i in train_index], [data[i] for i in test_index]
    train_labels, test_labels = [labels[i] for i in train_index], [labels[i] for i in test_index]

    # 임시 디렉토리 생성
    train_dir = 'tempo/train'
    val_dir = 'tempo/val'
    
    if not os.path.exists(train_dir):
        os.makedirs(os.path.join(train_dir, 'ILEUS'))
        os.makedirs(os.path.join(train_dir, 'NORMAL'))
    if not os.path.exists(val_dir):
        os.makedirs(os.path.join(val_dir, 'ILEUS'))
        os.makedirs(os.path.join(val_dir, 'NORMAL'))
    
    # 학습 데이터 복사
    for img, label in zip(train_data, train_labels):
        label_dir = 'ILEUS' if label == 0 else 'NORMAL'
        shutil.copy(img, os.path.join(train_dir, label_dir, os.path.basename(img)))
    
    # 검증 데이터 복사
    for img, label in zip(test_data, test_labels):
        label_dir = 'ILEUS' if label == 0 else 'NORMAL'
        shutil.copy(img, os.path.join(val_dir, label_dir, os.path.basename(img)))

    # 모델 학습
    model.train(data='tempo', 
                epochs=10,
                imgsz=224, 
                project=output_directory, 
                name=f"ileus_detection_fd{fold_idx+1}_", 
                save=True)

    # 모델 검증
    results = model.val()
    # 임시 디렉토리 삭제
    shutil.rmtree('tempo')
    fold_idx += 1
