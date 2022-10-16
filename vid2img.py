import os

import cv2

''' cut frames from videos to make dataset'''
vid_root = 'data/video2/'  # Path that include videos
save_path = 'data/crop_image/'

for vid in os.listdir(vid_root):
    print('vid', vid)
    vid_path = os.path.join(vid_root, vid)
    cap = cv2.VideoCapture(vid_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print ('fps: {}, frame_count: {}'.format(fps,frame_count))
    i = 0
    count = 0
    frame = 0
    while (cap.isOpened()):
        # Image preprocessing
        ret, img = cap.read()
        i += 1
        frame += 1
        if i == int (fps*1.3): # catch frame at each 1.5s

            cv2.imshow('img', img)
            cv2.imwrite(save_path + str(vid[:-4]) + '_' + 'new_' +str(count) + '.jpg', img) #save image
            count += 1
            i = 0
            cv2.waitKey(1)
        if frame > int(frame_count * 0.95): # for the loop to work properly
            break

