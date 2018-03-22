import datetime
import os.path as osp
import sys

import cv2

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(CURRENT_DIR)

from SiamFC.SiameseTracker import SiameseTracker
from darkflow.darkflow.net.build import TFNet

cap = cv2.VideoCapture(sys.argv[1])
# cap = cv2.VideoCapture(0)

options = {"model": "./darkflow/cfg/tiny-yolo-voc.cfg", "load": "./darkflow/tiny-yolo-voc.weights", "threshold": 0.3}
# options = {"model": "./cfg/yolo.cfg", "load": "./yolo.weights", "threshold": 0.3}
tracker_type = 'KCF'

tfnet = TFNet(options)
siamFC = SiameseTracker(debug=0,
                        checkpoint='SiamFC/Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained')
is_target_exist = False
cnt = 0
time_per_frame = 0


def get_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    return tracker


def preprocess(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return res
    # return cv2.resize(img, (800, 600))


def postprocess(img):
    res = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return res


while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = preprocess(frame)
    start_time = datetime.datetime.now()

    if is_target_exist:
        # ok, bbox = tracker.update(frame)
        bbox = siamFC.track(frame)
        # Draw bounding box
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    else:
        results = tfnet.return_predict(frame) if not is_target_exist else []
        is_contain_person = False
        maximum_index = 0
        maximum_confidence = -1
        for i, res in enumerate(results):
            if res['label'] == 'person':
                is_contain_person = True
                if res['confidence'] > maximum_confidence:
                    maximum_confidence = res['confidence']
                    maximum_index = i

        if is_contain_person:
            res = results[maximum_index]
            bbox = (res['topleft']['x'], res['topleft']['y'],
                    res['bottomright']['x'] - res['topleft']['x'],
                    res['bottomright']['y'] - res['topleft']['y'])
            # bbox = cv2.selectROI(frame, False)
            is_target_exist = True
            # tracker = get_tracker(tracker_type)
            # ok = tracker.init(frame, bbox)
            ok = siamFC.set_first_frame(frame, bbox)
            print(res)
            cv2.rectangle(frame, (res['topleft']['x'], res['topleft']['y']),
                          (res['bottomright']['x'], res['bottomright']['y']),
                          (0, 0, 255), 3)
            cv2.putText(frame, res['label'] + ':' + str(round(res['confidence'], 2)),
                        (res['topleft']['x'], res['topleft']['y'] - 12),
                        0, 1, (0, 0, 255), 1)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    time_per_frame = 0.9 * time_per_frame + 0.1 * duration.microseconds
    cv2.putText(frame, 'FPS ' + str(round(1e6 / time_per_frame, 1)),
                (30, 50), 0, 1, (0, 0, 255), 3)

    cv2.imshow('tracky', postprocess(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        is_target_exist = False

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
