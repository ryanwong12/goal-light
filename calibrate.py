# calibrate.py
import cv2

img = cv2.imread("dev/frames/frame_015780.jpg")
roi = cv2.selectROI("Select score bug region", img)
print(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
# Paste this output into config.py