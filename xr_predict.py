import tkinter as tk
from tkinter import Toplevel, Label, Button, messagebox, IntVar, Radiobutton, filedialog
from PIL import Image, ImageTk, ImageGrab
from io import BytesIO
from copy import deepcopy
import tempfile
import os
import sys
import atexit  # 종료 시 실행할 함수를 등록하기 위한 모듈
from ultralytics import YOLO


######################### FUNCTION ###########################

def resource_path(relative_path):
    """ Get the absolute path to the resource, whether we are running in a PyInstaller bundle or not. """
    try:
        # PyInstaller로 패키징된 환경에서는 _MEIPASS 경로를 사용합니다.
        base_path = sys._MEIPASS
    except Exception:
        # 개발 환경에서는 현재 스크립트의 디렉토리 경로를 사용합니다.
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



#### 모드 선택
def radiFunc():
    global selected_path

    if var.get() == 1:
        selected_path = selected_path = resource_path('best-pmn.pt')
    elif var.get() == 2:
        selected_path = selected_path = resource_path("best-pnx.pt")
    elif var.get() == 3:
        selected_path = selected_path = resource_path('best-ileus.pt')
    else:
        selected_path = resource_path('best-pnperi.pt')


#### 이미지 리사이즈 함수
def resize_image(image, max_width=500, max_height=600):
    """
    이미지를 지정된 최대 가로, 세로 크기를 넘지 않도록 리사이즈합니다.
    :param image: PIL 이미지 객체
    :param max_width: 최대 가로 크기 (기본값: 500)
    :param max_height: 최대 세로 크기 (기본값: 600)
    :return: 리사이즈된 PIL 이미지 객체
    """
    width, height = image.size

    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        return image

#### 클립보드에서 이미지를 가져와 Tkinter 창에 표시하는 함수
def show_image_from_clipboard():
    global photo
    global copied_image

    # 클립보드에서 이미지 가져오기
    clipboard_image = ImageGrab.grabclipboard()

    if clipboard_image and isinstance(clipboard_image, Image.Image):
        # 필요시 이미지를 리사이즈
        clipboard_image = resize_image(clipboard_image)
        copied_image = deepcopy(clipboard_image)

        # 이미지 크기 가져오기
        width, height = clipboard_image.size
        print(f"Image size: {width}x{height}")

        # 창 크기 조정
        root.geometry(f"{width + 20}x{height + 400}")

        # Label 크기 조정
        label.config(width=width, height=height)

        # 이미지 정보를 photo에 저장
        photo = ImageTk.PhotoImage(clipboard_image)
        label.configure(image=photo)
        label.image = photo  # 참조 유지

    else:
        messagebox.showerror("오류", "클립보드에서 이미지를 가져오는 데 실패했습니다.")


#### 클립보드에서 이미지를 바이트 스트림으로 변환
def convert_image_to_bytes(image):
    try:
        # 이미지가 RGBA 모드인 경우 RGB로 변환
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # 이미지를 JPEG 포맷으로 변환하여 BytesIO에 저장
        with BytesIO() as output:
            image.save(output, format='JPEG')
            jpg_data = output.getvalue()
            return jpg_data

    except Exception as e:
        messagebox.showerror(f"이미지 변환 중 오류 발생: {e}")
        return None


#### 이미지 예측 및 Tkinter 창에 표시하는 함수
def predict_and_display_image():
    global copied_image

    # 클립보드에서 이미지를 바이트 스트림으로 변환
    jpg_stream = convert_image_to_bytes(copied_image)

    if jpg_stream:
        try:
            # 임시 파일로 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(jpg_stream)
            temp_file.close()

            # YOLOv8n-cls 모델 로드
            model_path = selected_path
            model = YOLO(model_path)

            # YOLOv8n-cls 모델로 예측
            results = model(temp_file.name)

            # 결과를 Tkinter 창에 표시
            display_predicted_images(results)

            # 임시 파일 삭제
            os.remove(temp_file.name)

        except Exception as e:
            if var.get() != 1 and var.get() != 2 and var.get() != 3:
                messagebox.showerror("오류", "모드를 먼저 선택하세요")
            else:
                messagebox.showerror("오류", f"모델 예측 중 오류 발생: {e}")

    else:
        messagebox.showerror("오류", "이미지 변환 중 오류 발생")


#### 예측된 이미지를 Tkinter 창에 표시하는 함수
def display_predicted_images(results):
    global temp_image_path
    temp_image_path = os.path.abspath(os.path.dirname(__file__)) + "/temp19549851231.jpg"

    # 새 창 생성 및 설정
    new_window = Toplevel(root)
    new_window.title("이미지 예측 결과")

    try:
        for result in results:
            probs = result.probs  # Probs object for classification outputs
            result.save(filename=temp_image_path)  # Save to disk

        # 이미지를 Tkinter에서 사용할 수 있는 형식으로 변환
        tk_image = ImageTk.PhotoImage(file=temp_image_path)

        # Label에 이미지 표시
        label = Label(new_window, image=tk_image)
        label.image = tk_image  # 참조 유지
        label.pack()

        # 이미지 저장 함수
        def save_image():
            # 파일 저장 대화상자 열기
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", 
                                                     filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
            if save_path:
                try:
                    # 임시 이미지 파일을 지정된 경로에 저장
                    with open(temp_image_path, 'rb') as src, open(save_path, 'wb') as dst:
                        dst.write(src.read())
                    messagebox.showinfo("성공", "이미지가 성공적으로 저장되었습니다.")
                except Exception as e:
                    messagebox.showerror("오류", f"이미지 저장 중 오류 발생: {e}")

        # 이미지 저장 버튼 추가
        save_button = Button(new_window, text="이미지 저장", command=save_image)
        save_button.pack(pady=10)

    except Exception as e:
        messagebox.showerror("오류", f"예측 결과 표시 중 오류 발생: {e}")


#### 프로그램 종료 시 파일 삭제
def cleanup_temp_file():
    try:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print(f"Deleted temp file: {temp_image_path}")
    except Exception as e:
        print(f"Error deleting temp file: {e}")

# atexit를 사용하여 프로그램 종료 시 파일 삭제 함수 등록
atexit.register(cleanup_temp_file)



##########################################################
######################### MAIN ###########################
##########################################################

# 전역 변수설정
photo = None
copied_image = None
selected_path = None
temp_image_path = None  # 임시 파일 경로를 저장하기 위한 변수

# Tkinter 창 초기 설정
root = tk.Tk()
root.title("스크린샷 X-ray 이미지 분류")

# 이미지 표시를 위한 Label 위젯
label = Label(root, text="캡쳐한 이미지를 가져와 주세요.", width=50, height=20, relief="solid")
label.pack(pady=20)

# 3가지 pnm, pnx, ileus 모드 중 선택
label1 = Label(root, text="1. pneumonia, pneumothorax, ileus 중 모드 선택. \n2. 스크린샷 모드(shift+윈도우키+s)에서 x-ray 부분만 선택. \n3. '캡쳐한 이미지 가져오기' 클릭 \n4. '예측 및 표시' 클릭 ", fg="black")
label2 = Label(root, text="예측 완료 후 좌측 상단에 분류 확률이 0.0 ~ 1.0 사이로 표시됩니다.", fg="blue")

var = IntVar()
radi1 = Radiobutton(root, text="pneumonia", variable=var, value=1, command=radiFunc)
radi2 = Radiobutton(root, text="pneumothorax", variable=var, value=2, command=radiFunc)
radi3 = Radiobutton(root, text="ileus", variable=var, value=3, command=radiFunc)
radi4 = Radiobutton(root, text="pneumoperitonium(거의 예측 안됨)", variable=var, value=4, command=radiFunc)

label1.pack(pady=10)
label2.pack(pady=10)
radi1.pack()
radi2.pack()
radi3.pack()
radi4.pack()

# 버튼 클릭 시 클립보드에서 이미지 가져오는 버튼
button = Button(root, text="캡쳐한 이미지 가져오기", command=show_image_from_clipboard)
button.pack(pady=10)

# 버튼 클릭 시 display된 이미지 예측 및 표시
button = Button(root, text="예측 및 표시", command=predict_and_display_image)
button.pack(pady=10)

# Tkinter 메인 루프 실행
root.mainloop()


