import cv2
import paddlehub as hub
import cv2.data as datadir


print(datadir.haarcascades+'haarcascade_mcs_nose.xml')

# 尺寸
ds_factor = 1.25
# 开启摄像头
cap = cv2.VideoCapture(0)
if cap is None:
    raise IOError("Cannot open the webcam!")

nose_cascade = cv2.CascadeClassifier(datadir.haarcascades+'haarcascade_mcs_nose.xml')
if nose_cascade.empty():
    raise IOError('Unable to load the nose cascade classifier xml file')

module = hub.Module(name = "pyramidbox_lite_server_mask")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi',fourcc, 30, (640,480))

while True:
    ret, frame = cap.read()
    input_dict = {"data": [frame]}
    results = module.face_detection(data = input_dict)
    print(results)
    for r in results:
        lf = int(r['data']['left'])
        rt = int(r['data']['right'])
        top = int(r['data']['top'])
        bottom = int(r['data']['bottom'])
        wide = rt - lf
        high = bottom - top
        if r['data']['label'] == 'MASK':
            nose_rects = nose_cascade.detectMultiScale(frame[top:bottom, lf:rt], 1.05, 5, 0, (int(wide/6), int(high/6)), (int(wide/2), int(high/2)))
            # cv2.rectangle(frame, (lf, top), (rt, bottom), (0, 255, 0), 3)
            if len(nose_rects) == 0:
                cv2.rectangle(frame, (lf, top), (rt, bottom), (0, 255, 0), 3)
                cv2.putText(frame, r['data']['label'], (lf - 50, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),
                            2)
            else:
                # for (x_nose, y_nose, w_nose, h_nose) in nose_rects:
                #     cv2.rectangle(frame, (lf + x_nose, top + y_nose), (lf + x_nose + w_nose,top + y_nose + h_nose),
                #                   (255, 0, 0), 3)
                cv2.rectangle(frame, (lf, top), (rt, bottom), (0, 255, 255), 3)
                cv2.putText(frame, 'Wrong', (lf - 50, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255),
                            2)
        else:
            cv2.rectangle(frame, (lf, top), (rt, bottom), (0, 0, 255), 3)
            cv2.putText(frame, r['data']['label'], (lf - 50, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        # break

    cv2.imshow('Nose', frame)
    # frame = cv2.flip(frame, 0)  # 沿x轴翻转
    out.write(frame)
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
