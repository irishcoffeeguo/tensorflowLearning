import cv2, os
import paddlehub as hub
import cv2.data as datadir

print(datadir.haarcascades+'haarcascade_mcs_nose.xml')


nose_cascade = cv2.CascadeClassifier(datadir.haarcascades+'haarcascade_mcs_nose.xml')
if nose_cascade.empty():
    raise IOError('Unable to load the nose cascade classifier xml file')

module = hub.Module(name = "pyramidbox_lite_server_mask")

imglist = os.listdir('./imgs')

for img in imglist:
    theimg = os.path.join('./imgs', img)
    frame = cv2.imread(theimg)

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
            try:
                nose_rects = nose_cascade.detectMultiScale(frame[top:bottom, lf:rt], 1.05, 5, 0, (int(wide/6), int(high/6)), (int(wide/2), int(high/2)))
            except:
                nose_rects = []
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
        cv2.imwrite(os.path.join('./imgaftertest', img), frame)

    # cv2.imshow('face', frame)

    # cv2.waitKey(0)

