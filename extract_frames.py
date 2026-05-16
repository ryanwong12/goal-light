# extract_frames.py — run once per video
import cv2, os

sample_video_path = "samples/mtl_ott_110326.mp4"
cap = cv2.VideoCapture(sample_video_path)
os.makedirs("dev/frames", exist_ok=True)
frame_num = 0

video_fps = 30
fps = video_fps / 6  # Save 1 frame every 6 frames (5 fps) to reduce total frames and focus on key moments

while True:
    ret, frame = cap.read()
    if not ret:
        # print("Failed to read frame")
        break
    # Save 1 frame per second (not every frame — you'd get thousands)
    if frame_num % int(fps) == 0:  # assumes 30fps, adjust if needed
        cv2.imwrite(f"dev/frames/frame_{frame_num:06d}.jpg", frame)
    frame_num += 1

print(f"Extracted {frame_num // int(fps)} frames from the video.")