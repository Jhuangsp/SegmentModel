import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Video Segment DEMO', fromfile_prefix_chars='@')

parser.add_argument('-v', '--video', type=str, default='D:\\Astor\\Download\\dataset\\rotated\\rotated_squat_front.mp4',
                    help='Path to video (default:\"D:\\Astor\\Download\\dataset\\rotated\\rotated_squat_front.mp4\")')
parser.add_argument('-p', '--peaks', type=str, default='./model/peaks.npy',
                    help='Path to video (default:\"./model/peaks.npy\")')
args = parser.parse_args()

peaks = np.load(args.peaks)
peaks = np.pad(peaks, (1,0), 'constant', constant_values=0)
print('Peaks: {}'.format(list(peak for peak in peaks)))


cap = cv2.VideoCapture(args.video)
total_frame=cap.get(cv2.CAP_PROP_FRAME_COUNT)
print ("Total frame : {}".format(total_frame))

current = 0
now = 0
while 1:
    ret, frame = cap.read()
    if not ret:
        print("END")
        break

    cv2.putText(frame, 'frame:{}, video:{}'.format(current, now), (10, 80), cv2.FONT_HERSHEY_PLAIN,
                6, (255, 0, 0), 4, cv2.LINE_AA)
    cv2.imshow('frame', cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4)))
    if current in peaks[1:]:
        key = cv2.waitKey(0)
        if key == ord('q') or key == ord('Q'):
            print('[Q] Quit')
            break
        elif key == ord('r') or key == ord('R'):
            now = now
            current = peaks[now]+1
            cap.set(cv2.CAP_PROP_POS_FRAMES, current)
            print('[R] Replay video:{}'.format(now))
        elif key == ord('b') or key == ord('B'):
            now = max(now-1, 0)
            current = peaks[now]+1
            cap.set(cv2.CAP_PROP_POS_FRAMES, current)
            print('[B] Back to video:{}'.format(now))
        else:
            now += 1
            current+=1
            print('[ ] Next video:{}'.format(now))
    else:
        key = cv2.waitKey(30)
        current+=1
        
