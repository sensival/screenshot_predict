from ultralytics import YOLO
import os
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Pretrained YOLO model 준비
model = YOLO("yolov8n-cls.pt")

# 모델 save할 디렉토리
output_directory = "C:/Users/wogns/OneDrive/바탕 화면/dl/output"
# 데이터셋 디렉토리
dataset_directory = "C:/Users/wogns/OneDrive/바탕 화면/dl/pneumothorax"
# 데이터셋 디렉토리안에 있는 test 폴더 경로 설정
test_folder = "C:/Users/wogns/OneDrive/바탕 화면/dl/pneumothorax/test"


#---------------train--------------------

model.train(data=dataset_directory, 
            epochs=1, 
            imgsz=224, 
            project=output_directory, 
            name="pneumothorax_detection", 
            save=True,
            split=0.1) # validation data가 따로 없어서 split


#---------------validation------------------

# validation data가 따로 없음
# result_val = model.val()
# print("**** validation **** \n", result_val)


#----------------test--------------------
"""
print("**** test **** \n")

# 테스트 데이터 불러오기
test_images = []
test_labels = []

for label in ['NORMAL', 'PNEUMOTHORAX']:
    class_path = os.path.join(test_folder, label)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        test_images.append(img_path)
        test_labels.append(label)

# 배치 인퍼런스 수행
results_test = model(test_images)

# 예측 레이블 수집
test_preds = [result.probs.top1 for result in results_test]  # 가장 높은 확률을 가진 클래스 인덱스 선택
test_cls = [['NORMAL', 'PNEUMOTHORAX'][i] for i in test_preds]

# 정확도 계산
accuracy = accuracy_score(test_labels, test_cls)
print(f"Test Accuracy: {accuracy:.4f}")

#for (a, b) in zip(test_labels, test_preds):
#    print("label test:",a,['NORMAL', 'PNEUMONIA'][b])


# 분류 보고서 생성
report = classification_report(test_labels, test_cls, target_names=['NORMAL', 'PNEUMOTHORAX'])
print("\nClassification Report:\n", report)
with open(output_directory+'/classification_report_pnx.txt', 'w') as f:
    f.write(report)

# 혼동 행렬 생성
cm = confusion_matrix(test_labels, test_cls)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'pneumthorax'], yticklabels=['normal', 'pneumthorax'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(output_directory+'/confusion_matrix_pnx.png')

"""


#----------------predict--------------------
"""
# Run batched inference on a list of images
results_img = model(["chest.jpg"])

# Process results list
for result in results_img:
    probs = result.probs  # Probs object for classification outputs
    print(f"Class probabilities: {probs}")
    top1_index = probs.top1  # 가장 높은 확률을 가진 클래스의 인덱스
    print(f"Top1 class index: {top1_index}")
    result.show()  # Display to screen
    result.save(filename="result.jpg")  # Save to disk
"""