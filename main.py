import cv2

cap = cv2.VideoCapture(0)
img_counter = 0
video_counter = 0
rotate_right = 0
rotate_left = 0
record = False

print("start stream")

while True:
    check, frame = cap.read()

    if frame is None:
        print("No frame")
        cap.stop()

    k = cv2.waitKey(1) & 0xFF

    res = cv2.resize(frame, (700, 500))
    cropped_image = res[200:400, 200:400]

    if k == ord('i'):
        #Вывод в отдельное окно часть экрана равное cropped_image
        cv2.imwrite("roi_{}.png".format(img_counter), cropped_image)
        img_counter += 1

    if k == ord('r'):
        #Запись с экрана до нажатия s
        record = cv2.VideoWriter("out_{}.mp4".format(video_counter), cv2.VideoWriter_fourcc(*'mp4v'), 30, cropped_image.shape[:2])
        video_counter += 1
        print('recording')
    if record:
        record.write(cropped_image)
        if k == ord('s'):
            print('stop')
            record.release()
            record = False

    if k == ord('l'):
        #Выбор и выведение ROI в отдельное окно
        r = cv2.selectROI("area", res)
        cropped_image2 = res[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        cv2.imshow("crop", cropped_image2)

    fps = cap.get(cv2.CAP_PROP_FPS)

    #Показывает фпс видео
    cv2.putText(res, str(fps), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

    if k == ord('d'):
        #Поворот видео направо
        rotate_right += 1

    if rotate_right == 1:
        res = cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_right == 2:
        res = cv2.rotate(res, cv2.ROTATE_180)
    elif rotate_right == 3:
        res = cv2.rotate(res, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate_right == 4:
        rotate_right = 0

    if k == ord('a'):
        # Поворот видео налево
        rotate_left += 1

    if rotate_left == 1:
        res = cv2.rotate(res, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate_left == 2:
        res = cv2.rotate(res, cv2.ROTATE_180)
    elif rotate_left == 3:
        res = cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_left == 4:
        rotate_left = 0

    if k == ord('j'):
        print(res.shape[:2])

    cv2.imshow('out', res)
