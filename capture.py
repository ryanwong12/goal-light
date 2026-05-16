import cv2

cap = cv2.VideoCapture(0)  # or /dev/video0 on Linux

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("feed", frame)
    if cv2.waitKey(1) == ord("q"):
        break
