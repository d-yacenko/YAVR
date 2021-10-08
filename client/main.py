import os
import cv2
import numpy as np
import requests
import pyfakewebcam
import PIL
from PIL import Image
import io
from io import BytesIO
import time
import sys
import re
from multiprocessing import Process, Queue

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if len(sys.argv)<8:
    print(" <cam_in> <cam_out> <background> <resolution> <server_url> <edfe_soft_mask> <holgram_effect>  arguments missing!")
    print("python3 main.py /dev/video0 /dev/video22 background6.jpg 720x1280 http://192.168.49.110:8080")
    sys.exit(-1)
cam_in=sys.argv[1]
cam_out=sys.argv[2]
cam_background=sys.argv[3]
cam_resolution=sys.argv[4]
cam_server_url=sys.argv[5]
cam_edfe_soft_mask=str2bool(sys.argv[6])
cam_holgram_effect=str2bool(sys.argv[7])

TIMEOUT=0.03
#H,W = 720,1280
H,W =int(re.findall(r'\d+',cam_resolution)[0]),int(re.findall(r'\d+',cam_resolution)[1])

img_prev=Image.new(mode="L", size=(W,H))
Q=Queue()

def get_mask(frame, bodypix_url=cam_server_url):
    _, data = cv2.imencode(".jpg", frame)
    global img_prev
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    #img.save('234.jpeg')
    byte_io = BytesIO()
    img.save(byte_io, 'jpeg')
    byte_io.seek(0)
    file=byte_io.getvalue()
    action ={"file":file,"name":'file'}
    try:
        r = requests.post(url=bodypix_url,files=action,timeout=0.05)
        img = Image.open(io.BytesIO(r.content))
        img=img.convert('L')
        img_prev=img
        #img.save('test.jpg')
    except:
        #print("missing responce")
        img=img_prev
    mask =  np.asarray(img)
    return mask

def post_process_mask(mask):
    mask = cv2.dilate(mask, np.ones((10,10), np.uint8) , iterations=1)
    mask = cv2.blur(mask.astype(float), (30,30))
    return mask

def shift_image(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy>0:
        img[:dy, :] = 0
    elif dy<0:
        img[dy:, :] = 0
    if dx>0:
        img[:, :dx] = 0
    elif dx<0:
        img[:, dx:] = 0
    return img

def hologram_effect(img):
    # окрашиваем в синий оттенок
    holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
    # добавляем эффект полутонов
    bandLength, bandGap = 2, 3
    for y in range(holo.shape[0]):
        if y % (bandLength+bandGap) < bandLength:
            holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)
    # эффект привидения
    holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
    holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)
    # комбинируем с перенасыщенным исходным цветом
    out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
    return out

def get_frame(cap, background_scaled):
    _, frame = cap.read()
    # загружаем маску с поддержкой повторов (приложению нужно разогреться, а мы ленивы)
    # всё будет согласовано по прошествии некоторого времени
    mask = None
    while mask is None:
        try:
            mask = get_mask(frame)
        except requests.RequestException:
            print("mask request failed, retrying")
    # пост-процессинг маски и кадра
    mask_=mask.copy()
    idx = np.where(mask != 0)
    mask_[idx] = 1
    mask=mask_
    if cam_edfe_soft_mask: mask = post_process_mask(mask)
    if cam_holgram_effect: frame = hologram_effect(frame)
# комбинируем фон и передний план
    inv_mask = 1-mask
    for c in range(frame.shape[2]):
        frame[:,:,c] = frame[:,:,c]*mask + background_scaled[:,:,c]*inv_mask
#        frame[:,:,c] = frame[:,:,c] + background_scaled[:,:,c]
    return frame

# настраиваем доступ к реальной камере
cap = cv2.VideoCapture(cam_in)
height, width = H,W
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 30)

# настраиваем фиктивную камеру
fake = pyfakewebcam.FakeWebcam(cam_out, width, height)

video=True

# загружаем новый виртуальный фон
if not cam_background.endswith('.mp4'):
    background = cv2.imread(cam_background)
    background_scaled = cv2.resize(background, (width, height))
    video=False
else:
    cap1 = cv2.VideoCapture(cam_background)
    height, width = H,W
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap1.set(cv2.CAP_PROP_FPS, 30)
    video=True


# вечный цикл перебора кадров
while True:
    #start=time.time()
    if video:
         ret, background_scaled = cap1.read()
         if  ret==False:
             cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
             _, background_scaled = cap1.read()
    frame = get_frame(cap, background_scaled)
    #end=time.time()
    #print(f"Runtime of the program is {end - start}")
    # фиктивная камера ожидает RGB-изображение
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fake.schedule_frame(frame)
