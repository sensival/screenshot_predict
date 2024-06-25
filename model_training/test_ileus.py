from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ultralytics import YOLO
import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

dataset_path = 'C:/Users/wogns/OneDrive/바탕 화면/dl/ileus/test'
ileus_path = os.path.join(dataset_path, 'ILEUS')
normal_path = os.path.join(dataset_path, 'NORMAL')
output_directory = "C:/Users/wogns/OneDrive/바탕 화면/dl/output/v1_ileus_test"
now_dir = os.path.abspath(os.path.dirname(__file__))

modeldir =["C:/Users/wogns/OneDrive/바탕 화면/dl/ileuspt/fd1.pt",
            "C:/Users/wogns/OneDrive/바탕 화면/dl/ileuspt/fd2.pt",
           "C:/Users/wogns/OneDrive/바탕 화면/dl/ileuspt/fd3.pt",
           "C:/Users/wogns/OneDrive/바탕 화면/dl/ileuspt/fd4.pt",
           "C:/Users/wogns/OneDrive/바탕 화면/dl/ileuspt/fd5.pt",
           "C:/Users/wogns/OneDrive/바탕 화면/dl/ileuspt/fd6.pt",
           "C:/Users/wogns/OneDrive/바탕 화면/dl/ileuspt/fd7.pt",
           "C:/Users/wogns/OneDrive/바탕 화면/dl/ileuspt/fd8.pt",
           "C:/Users/wogns/OneDrive/바탕 화면/dl/ileuspt/fd9.pt",
           "C:/Users/wogns/OneDrive/바탕 화면/dl/ileuspt/fd10.pt"]


def add_text_to_image(image_path, text, output_path):
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # 폰트 설정
        try:
            font = ImageFont.truetype("arial.ttf", 12)  # 폰트 경로가 맞지 않으면 기본 폰트로 대체
        except IOError:
            font = ImageFont.load_default()

        # 텍스트 크기 계산
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # 텍스트 위치 계산
        width, height = img.size
        text_x = width - text_width - 10
        text_y = height - text_height - 10

        # 이미지에 텍스트 추가
        draw.text((text_x, text_y), text, (255, 0, 0), font=font)
        img.save(output_path)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


fold_idx = 0
for dir in modeldir:
    # 성능 평가 및 출력
    print("**** test **** \n")
    model = YOLO(dir)

    # 테스트 데이터 불러오기
    test_images = []
    test_labels = []

    for label in ['ILEUS', 'NORMAL']:
        class_path = os.path.join(dataset_path, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            test_images.append(img_path)
            test_labels.append(label)

    # 배치 인퍼런스 수행
    results_test = model(test_images)

    # 예측 레이블 수집
    test_preds = [result.probs.top1 for result in results_test]  # 가장 높은 확률을 가진 클래스 인덱스 선택
    test_cls = [ ['ILEUS', 'NORMAL'][i] for i in test_preds]

    # 정확도 계산
    accuracy = accuracy_score(test_labels, test_cls)
    print(f"Test Accuracy: {accuracy:.4f}")

    #for (a, b) in zip(test_labels, test_preds):
    #    print("label test:",a, ['ILEUS', 'NORMAL'][b])


    # 분류 보고서 생성
    report = classification_report(test_labels, test_cls, target_names= ['ILEUS', 'NORMAL'])
    print("\nClassification Report:\n", report)
    with open(output_directory+f'/classification_report_ileus_fd{fold_idx+1}.txt', 'w') as f:
        f.write(report)

    # 혼동 행렬 생성
    cm = confusion_matrix(test_labels, test_cls)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ileus','normal'], yticklabels=['ileus','normal'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_directory+f'/confusion_matrix_ileus_fd{fold_idx+1}.png')


    for img_path, true_label, pred_label in zip(test_images, test_labels, test_cls):
        pred_label_text = f"predict: {pred_label}"
        true_label_text = f"actual: {true_label}"
        output_image_path = os.path.join(output_directory, f"{os.path.basename(img_path)}_pred_fd{fold_idx+1}.png")

        # 텍스트 추가
        add_text_to_image(img_path, f"{true_label_text} {pred_label_text}", output_image_path)

    print(f"모델 {fold_idx+1}의 예측 완료 및 결과 저장")
    fold_idx += 1    
