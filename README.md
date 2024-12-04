## 🌡 Screenshot predict 소개

><b>사용법</b><br>1. pneumonia, pneumothorax, ileus, pneumothorax 중 모드 선택. <br>2. 스크린샷 모드(shift+윈도우키+s)에서 x-ray 부분만 선택. <br>3. '캡쳐한 이미지 가져오기' 클릭 <br>4. '예측 및 표시' 클릭 <br>5. 예측 완료 후 좌측 상단에 분류 확률이 0.0 ~ 1.0 사이로 표시되며 필요시 저장가능

<br/>
<img width="850" alt="실행화면" src="https://github.com/sensival/screenshot_predict/assets/136985426/cc2d919c-48f3-489f-951a-95d7be6c8547">


<hr>

## 💌 Contributor

**`Sole contributer`**

<hr>


## ⚒️ Development Tools

<br/>
<div align="left">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
  <img src="https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white"/>
</div\>

<br/>
<br>

><b> Pretrained model: </b>YOLOv8n-cls(https://docs.ultralytics.com/ko/models/yolov8/)<br><b>GUI: </b>tkinter(https://docs.python.org/3/library/tkinter.html)

<hr>

## 📝 Dataset 

><b> Pneumonia dataset: </b>https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia<br><b>Pneumothorax dataset: </b>https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks<br><b>Ileus dataset: </b>Manual scraping and personally labeling the data <br><b>Pneumoperitoneum dataset: </b>Manual scraping and personally labeling the data <br>

|  | Pneumonia | Pneumothorax | Ileus | Pneumothorax |
| :-: |  :-: | :-: | :-: | :-: |
| Train | Disease: 3875 images <br> Normal: 1341 images | Disease: 2379 images <br> Normal: 8296 images<br>  |  Disease: 51 images <br> Normal: 138 images <br> | Disease: 25 images <br> Normal: 188 images <br> |
| Validation | Disease: 8 images <br> Normal: 8 images | The train data was divided(0.1)|  K-fold-cross-validation(k=10) | K-fold-cross-validation(k=10) |
| Test | Disease: 390 images <br> Normal: 234 images | Disease: 390 images <br> Normal: 234 images|  Disease: 16 images <br> Normal: 11 images | Disease: 5 images <br> Normal: 11 images |

<hr>

## 👋 About YOLOv8
> YOLOv8 is the latest iteration in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed. The YOLOv8 series offers a diverse range of models, each specialized for specific tasks in computer vision. Additionally, these models are compatible with various operational modes including Inference, Validation, Training, and Export, facilitating their use in different stages of deployment and development.<br>

<img width="850" alt="YOLO성능" src="https://github.com/sensival/screenshot_predict/assets/136985426/ef2720dd-0c26-4f68-bb41-47118a2ddbe2">

#### YOLOv8n-cls is a model that uses fewer resources, has fast computational speed, and relatively lower accuracy. (This model was chosen because it needs to be trained on a personal computer.)

<br/>
<hr>

## ✏️ Development Details

### Directory setting
The project's data directory is organized as follows:
- **pneumonia/**
  - **train/**
    - **NORMAL/**
    - **PNEUMONIA/**
  - **val/**
    - **NORMAL/**
    - **PNEUMONIA/**
  - **test/**
    - **NORMAL/**
    - **PNEUMONIA/**
- **pneumothorax/**
  - **train/**
    - **NORMAL/**
    - **PNEUMOTHORAX/**
  - **test/**
    - **NORMAL/**
    - **PNEUMOTHORAX/**
- **ileus/**
  - **train/**
    - **ILEUS/**
    - **NORMAL/**
  - **test/**
    - **ILEUS/**
    - **NORMAL/**
- **pnperi/** 
  - **train/**
    - **FREEAIR/** *'free air' : Pneumoperitoneum findings
    - **NORMAL/**  
  - **test/**
    - **FREEAIR/**
    - **NORMAL/**



### Train/Validation code

#### 1. Pneumonia

```python
model.train(data=dataset_directory, 
            epochs=100, 
            imgsz=224, 
            project=output_directory, 
            name="pneumonia_detection", 
            save=True)
```

#### 2. Pneumothorax
```python
model.train(data=dataset_directory, 
            epochs=100, 
            imgsz=224, 
            project=output_directory, 
            name="pneumothorax_detection", 
            save=True,
            split=0.1) # validation data가 따로 없어서 split
```
#### 3. Ileus/Pneumoperitoneum(k-fold)
```python
# ------- : ILEUS or FREEAIR
k = 10
kf = KFold(n_splits=k)

fold_idx = 0
for train_index, test_index in kf.split(data):
    # 각 폴드별로 데이터 분할
    train_data, test_data = [data[i] for i in train_index], [data[i] for i in test_index]
    train_labels, test_labels = [labels[i] for i in train_index], [labels[i] for i in test_index]

    # 임시 디렉토리 생성
    train_dir = 'tempo/train'
    val_dir = 'tempo/val'
    
    if not os.path.exists(train_dir):
        os.makedirs(os.path.join(train_dir, '-------'))
        os.makedirs(os.path.join(train_dir, 'NORMAL'))
    if not os.path.exists(val_dir):
        os.makedirs(os.path.join(val_dir, 'ILEUS'))
        os.makedirs(os.path.join(val_dir, 'NORMAL'))
    
    # 학습 데이터 복사
    for img, label in zip(train_data, train_labels):
        label_dir = '-------' if label == 0 else 'NORMAL'
        shutil.copy(img, os.path.join(train_dir, label_dir, os.path.basename(img)))
    
    # 검증 데이터 복사
    for img, label in zip(test_data, test_labels):
        label_dir = '-------' if label == 0 else 'NORMAL'
        shutil.copy(img, os.path.join(val_dir, label_dir, os.path.basename(img)))

    # 모델 학습
    model.train(data='tempo', 
                epochs=10,
                imgsz=224, 
                project=output_directory, 
                name=f"-------_detection_fd{fold_idx+1}_", 
                save=True)

    # 모델 검증
    results = model.val()

    # 임시 디렉토리 삭제
    shutil.rmtree('tempo')
    fold_idx += 1
```


### Test code
```python
# ------- : Condition name
# best 모델 dir
best_dir =()
model = YOLO("best_dir")

# 테스트 데이터 불러오기
test_images = []
test_labels = []

for label in ['NORMAL', '-------']:
    class_path = os.path.join(test_folder, label)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        test_images.append(img_path)
        test_labels.append(label)

results_test = model(test_images)

# 예측 레이블 수집
test_preds = [result.probs.top1 for result in results_test]  # 가장 높은 확률을 가진 클래스 인덱스 선택
test_cls = [['NORMAL', '-------'][i] for i in test_preds]

# 정확도 계산
accuracy = accuracy_score(test_labels, test_cls)
print(f"Test Accuracy: {accuracy:.4f}")

# 분류 보고서 생성
report = classification_report(test_labels, test_cls, target_names=['NORMAL', '-------'])
print("\nClassification Report:\n", report)
with open(output_directory+'/classification_report_-------.txt', 'w') as f:
    f.write(report)

# confusion_matrix 생성
cm = confusion_matrix(test_labels, test_cls)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', '-------'], yticklabels=['normal', '-------'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(output_directory+'/confusion_matrix_-------.png')
```

### GUI
[Refer to the source code](https://github.com/sensival/screenshot_predict/blob/main/xr_predict.py)<br>

### Result
#### Train/Validation
##### 1. Pneumonia
<img width="500" alt="pmn tr" src="https://github.com/sensival/screenshot_predict/assets/136985426/69e3fbc0-192a-4975-aa2d-d2f0b057ed0c"><br>

Due to the small size of the validation dataset, it was difficult to determine the point of overfitting, so it was trained for an additional 100 epochs.<br>
<img width="500" alt="pmn tr2" src="https://github.com/sensival/screenshot_predict/assets/136985426/9ce6e541-4732-4de5-933b-c9e8fefeffec"><br>
The validation results are still unstable. It needs to increase the size of the validation dataset in the future<br>

##### 2. Pneumothorax
<img width="500" alt="pnx tr" src="https://github.com/sensival/screenshot_predict/assets/136985426/4016dbf1-1370-4632-b25c-2d5d9ea7f2be"><br>

##### 3. Ileus/Pneumoperitoneum(the fold with the best test results among the 10 folds)
Ileus: fold 7<br>
<img width="500" alt="il tr" src="https://github.com/sensival/algorithm_practice/assets/136985426/005e21f9-5b00-40a3-a26d-aef60182282e"><br>
Pneumoperitoneum: fold 6<br>
<img width="500" alt="pnp tr" src="https://github.com/sensival/algorithm_practice/assets/136985426/8f2a3b25-741f-4987-bd45-ee0e6fbcd22c"><br>
This 2 models also have unstable train/validation loss due to the insufficient size of the train dataset.<br>

#### Test
##### 1. Pneumonia
<img width="550" alt="pmn tt" src="https://github.com/sensival/screenshot_predict/assets/136985426/4687a1da-81fb-4f4c-9814-4f9922430db3"><br>
<img width="450" alt="pmn tt" src="https://github.com/sensival/screenshot_predict/assets/136985426/d1ad3a79-3bb5-4af3-803b-f08e294fb48d"><br>

##### 2. Pneumothorax
<img width="550" alt="pnx tt" src="https://github.com/sensival/algorithm_practice/assets/136985426/72ce402c-ee37-4ee7-a201-a8ef2877c4d3"><br>
<img width="450" alt="pnx tt" src="https://github.com/sensival/algorithm_practice/assets/136985426/30668c92-37a8-4108-9f05-ce318e3136b7"><br>


##### 3. Ileus(fold 7)
<img width="550" alt="il tt" src="https://github.com/sensival/algorithm_practice/assets/136985426/7dc8362d-5db1-428a-ae9a-fc6f510a743e"><br>
<img width="450" alt="il tt" src="https://github.com/sensival/algorithm_practice/assets/136985426/ffde15ad-90fb-4e1d-8cc7-c68051eef1d8"><br>

##### 4. Pneumoperitoneum(fold 6)
<img width="550" alt="pp tt" src="https://github.com/sensival/algorithm_practice/assets/136985426/b71dc642-a3d3-4cb6-ac84-49be8a2d1d1e"><br>
<img width="450" alt="pp tt" src="https://github.com/sensival/algorithm_practice/assets/136985426/d1e3082f-ff95-4e7f-884d-856b9ca44213"><br>

<br/>
<hr/>
