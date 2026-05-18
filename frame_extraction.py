import cv2
import os

video_path = r'C:\Users\asus\Desktop\clamp detection\Task.mp4'
output_folder = 'frames'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Extract 30 frames evenly
frame_indices = [int(i * (total_frames / 30)) for i in range(30)]

count = 0
extracted = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count in frame_indices:
        cv2.imwrite(f"{output_folder}/clamp_frame_{extracted}.jpg", frame)
        extracted += 1
    count += 1

cap.release()
print(f"Extracted {extracted} frames.")