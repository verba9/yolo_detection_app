import streamlit as st
import torch
import tempfile
import cv2
from PIL import Image
import os
import json
import pandas as pd
from datetime import datetime

st.title("Мониторинг судов в порту (YOLOv5)")

# Загрузка модели
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

# Классы, относящиеся к судам (в COCO 'boat')
TARGET_CLASSES = ['boat']

# История
def save_history(filename, results, filtered_objs):
    entry = {
        "file": filename,
        "timestamp": datetime.now().isoformat(),
        "detected_boats": filtered_objs
    }
    try:
        with open("history.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"Ошибка при сохранении истории: {e}")

# Загрузка изображения или видео
file = st.file_uploader("\U0001F4E5 Загрузите изображение или видео (судоходного порта)", type=["jpg", "jpeg", "png", "mp4"])

if file:
    suffix = file.name.split('.')[-1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.' + suffix)
    temp_file.write(file.read())
    temp_path = temp_file.name

    if suffix in ['jpg', 'jpeg', 'png']:
        image = Image.open(temp_path)
        st.image(image, use_container_width=True)
        results = model(temp_path)
        df = results.pandas().xyxy[0]
        boats = df[df['name'].isin(TARGET_CLASSES)]
        results.render()
        st.image(results.ims[0], caption=f"Обнаружено судов: {len(boats)}", use_container_width=True)
        save_history(file.name, results, boats.to_dict(orient="records"))

    elif suffix == 'mp4':
        cap = cv2.VideoCapture(temp_path)
        stframe = st.empty()
        out_path = "results/output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        boat_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            df = results.pandas().xyxy[0]
            boats = df[df['name'].isin(TARGET_CLASSES)]
            boat_count += len(boats)
            img = results.render()[0]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if out is None:
                out = cv2.VideoWriter(out_path, fourcc, 20.0, (img.shape[1], img.shape[0]))
            out.write(img_bgr)
            stframe.image(img, channels="RGB")

        cap.release()
        out.release()
        st.video(out_path)
        save_history(file.name, results, [{"count": boat_count}])

# История и экспорт
if st.button("Показать историю"):
    # Проверка наличия файла history.json
    if os.path.exists("history.json"):
        try:
            with open("history.json", "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f.readlines()]
            df = pd.json_normalize(lines, record_path='detected_boats', meta=['file', 'timestamp'])
            st.dataframe(df)
            df.to_excel("detected_ships.xlsx", index=False)
            st.success("Отчёт сохранён как detected_ships.xlsx")
        except json.JSONDecodeError as e:
            st.error(f"Ошибка при чтении файла history.json: {e}")
    else:
        st.warning("Файл history.json не найден.")