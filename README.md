## 🌡 Screenshot predict 소개

><b>사용법</b><br>1. pneumonia, pneumothorax, ileus, pneumothorax 중 모드 선택. <br>2. 스크린샷 모드(shift+윈도우키+s)에서 x-ray 부분만 선택. <br>3. '캡쳐한 이미지 가져오기' 클릭 <br>4. '예측 및 표시' 클릭 <br>5. 예측 완료 후 좌측 상단에 분류 확률이 0.0 ~ 1.0 사이로 표시되며 필요시 저장가능

<br/>
<img width="850" alt="실행화면" src="https://github.com/sensival/screenshot_predict/assets/136985426/cc2d919c-48f3-489f-951a-95d7be6c8547">


<hr>

## 💌 Contributor

Development by: [@sensival](https://github.com/sensival/)<br>
Data Labeling by: Internal medicine R2 at SNUH

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
| Train | Disease: 3875 images <br> Normal: 1341  | Disease: 2379 images <br> Normal: 8296<br>  |  Disease: 51 images <br> Normal: 138<br> | Disease: 25 images <br> Normal: 188<br> |
| Validation | Disease: 8 images <br> Normal: 8  | The train data was divided(0.1)|  K-fold-cross-validation(k=10) | K-fold-cross-validation(k=10) |
| Test | Disease: 390 images <br> Normal: 234  | The train data was divided(0.1)|  Disease: 16 images <br> Normal: 11 | Disease: 5 images <br> Normal: 11 |

<hr>

## 👋 About YOLOv8



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
    - **NORMAL/**
    - **ILEUS/**
  - **test/**
    - **NORMAL/**
    - **ILEUS/**
- **pnperi/** 
  - **train/**
    - **NORMAL/**
    - **FREEAIR/** : 'free air' : Pneumoperitoneum findings
  - **test/**
    - **NORMAL/**
    - **FREEAIR/**

<hr/>


### Train/Validation code



### Test code



### GUI


# Result


<br/>
<hr/>
