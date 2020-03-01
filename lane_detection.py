import cv2
import numpy as np

cap = cv2.VideoCapture(r'C:\Users\Aditya\Pictures\test_video.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()

    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    h, w, c = frame.shape

    vertices = [(0, h),(w/2, h/2),(w, h)]
    vertices = np.array([vertices], np.int32)

    canny_image = cv2.Canny(gray_image, 100, 150)

    img = np.zeros_like(gray_image)
    match_mask_color = 255
    cv2.fillPoly(img, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(canny_image, img)

    cv2.imshow('canny', masked_image)

    lines = cv2.HoughLinesP(masked_image, rho=2, theta=np.pi/100, threshold=50, lines=np.array([]),
                            minLineLength = 40, maxLineGap = 100)


    frame1 = np.copy(frame)
    blank_image = np.zeros((h, w, c), dtype = np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), thickness = 10)

    lane_detection = cv2.addWeighted(frame1, 0.8, blank_image, 1, 0.0)

    cv2.imshow('canny', lane_detection)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




