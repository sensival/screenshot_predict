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
dataset_path = 'C:/Users/wogns/OneDrive/바탕 화면/dl/pnperi/train'
ileus_path = os.path.join(dataset_path, 'FREEAIR')
normal_path = os.path.join(dataset_path, 'NORMAL')
output_directory = "C:/Users/wogns/OneDrive/바탕 화면/dl/output"
now_dir = os.path.abspath(os.path.dirname(__file__))

# 이미지 파일 리스트 생성
ileus_images = [os.path.join(ileus_path, f) for f in os.listdir(ileus_path) if f.endswith('.jpg')]
normal_images = [os.path.join(normal_path, f) for f in os.listdir(normal_path) if f.endswith('.jpg')]

# 전체 데이터 리스트와 레이블 리스트 생성
data = ileus_images + normal_images
labels = [0] * len(ileus_images) + [1] * len(normal_images)  # FREEAIR = 0, NORMAL = 1

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
        os.makedirs(os.path.join(train_dir, 'FREEAIR'))
        os.makedirs(os.path.join(train_dir, 'NORMAL'))
    if not os.path.exists(val_dir):
        os.makedirs(os.path.join(val_dir, 'FREEAIR'))
        os.makedirs(os.path.join(val_dir, 'NORMAL'))
    
    # 학습 데이터 복사
    for img, label in zip(train_data, train_labels):
        label_dir = 'FREEAIR' if label == 0 else 'NORMAL'
        shutil.copy(img, os.path.join(train_dir, label_dir, os.path.basename(img)))
    
    # 검증 데이터 복사
    for img, label in zip(test_data, test_labels):
        label_dir = 'FREEAIR' if label == 0 else 'NORMAL'
        shutil.copy(img, os.path.join(val_dir, label_dir, os.path.basename(img)))

    # 모델 학습
    model.train(data='tempo', 
                epochs=10,
                imgsz=224, 
                project=output_directory, 
                name=f"pnperi_detection_fd{fold_idx+1}_", 
                save=True)

    # 모델 검증
    results = model.val()

    # 성능 평가 및 출력
    print("**** test **** \n")

    # 테스트 데이터 불러오기
    test_images = []
    test_labels = []

    for label in ['FREEAIR', 'NORMAL']:
        class_path = os.path.join(now_dir, 'tempo\\val', label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            test_images.append(img_path)
            test_labels.append(label)

    # 배치 인퍼런스 수행
    results_test = model(test_images)

    # 예측 레이블 수집
    test_preds = [result.probs.top1 for result in results_test]  # 가장 높은 확률을 가진 클래스 인덱스 선택
    test_cls = [ ['FREEAIR', 'NORMAL'][i] for i in test_preds]

    # 정확도 계산
    accuracy = accuracy_score(test_labels, test_cls)
    print(f"Test Accuracy: {accuracy:.4f}")

    #for (a, b) in zip(test_labels, test_preds):
    #    print("label test:",a, ['FREEAIR', 'NORMAL'][b])


    # 분류 보고서 생성
    report = classification_report(test_labels, test_cls, target_names= ['FREEAIR', 'NORMAL'])
    print("\nClassification Report:\n", report)
    with open(output_directory+f'/classification_report_pnperi_fd{fold_idx+1}.txt', 'w') as f:
        f.write(report)

    # 혼동 행렬 생성
    cm = confusion_matrix(test_labels, test_cls)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['free air','normal'], yticklabels=['free air','normal'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_directory+f'/confusion_matrix_pnperi_fd{fold_idx+1}.png')
        
    # 임시 디렉토리 삭제
    shutil.rmtree('tempo')
    
    
    fold_idx += 1


'''
from ultralytics import YOLO
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold
import numpy as np

# 기본 설정
data_dir = "C:/Users/wogns/OneDrive/바탕 화면/dl/ileus"  # 이미지 데이터셋 경로
output_directory = "C:/Users/wogns/OneDrive/바탕 화면/dl/output"  # 출력 디렉토리

# Load your pre-trained YOLOv8 classification model
model = YOLO('yolov8n-cls.pt')

# Example dataset
data = []
labels = []

# 라벨을 인덱스로 변환하기 위한 딕셔너리 생성
label_to_index = {'NORMAL': 0, 'ILEUS': 1}  # 예시로 라벨 인덱스 매핑


for label in ['NORMAL', 'ILEUS']:
    class_path = os.path.join(data_dir, label)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        data.append(img_path)
        labels.append(label_to_index[label])

# K-Fold Cross-Validation
kf = KFold(n_splits=10)
fold_idx = 0
for train_index, val_index in kf.split(data):
    fold_idx += 1
    train_data, val_data = data[train_index], data[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]
    
    # Train the model on the training data
    model.train(data={'train': train_data, 'val': val_data, 'labels': train_labels},
                epochs=10,
                imgsz=224,
                project=output_directory, 
                name=f"ileus_detection_fold{fold_idx}_", 
                save=True)
    
    # Validate the model on the validation data
    results = model.val(data={'val': val_data, 'labels': val_labels})
    print(f'Validation results: {results}')





# 추가로 실행하고 싶은 inference 예시
# Run batched inference on a list of images
inference_images = ["chest.jpg"]
results = model(inference_images)

# Process results list
for result in results:
    probs = result.probs  # Probs object for classification outputs
    result.show()  # Display to screen
    result.save(filename="result.jpg")  # Save to disk
'''
