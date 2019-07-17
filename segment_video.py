import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Video Segment DEMO', fromfile_prefix_chars='@')

parser.add_argument('-v', '--video', type=str, default='D:\\Astor\\Download\\dataset\\rotated\\rotated_squat_front.mp4',
                    help='Path to video (default:\"D:\\Astor\\Download\\dataset\\rotated\\rotated_squat_front.mp4\")')
parser.add_argument('-p', '--peaks', type=str, default='./model/peaks.npy',
                    help='Path to peaks.npy (default:\"./model/peaks.npy\")')
parser.add_argument('-d', '--display', action='store_true',
                    help='Display all fragments.')
parser.add_argument('-s', '--store', action='store_true',
                    help='Store all fragments to tiny videos.')
args = parser.parse_args()

 
def convert_frames_to_video(cap,pathOut,fps,cut,size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    current = 0

    for tiny in range(len(cut)):
        print('Writing tiny video {}'.format(tiny))

        out = cv2.VideoWriter('{}_{}.avi'.format(pathOut, tiny),cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
        while 1:
            ret, frame = cap.read()
            if ret:
                out.write(cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4)))
                current+=1
                if current in peaks:
                    break
            else:
                break
        out.release()
 

peaks = np.load(args.peaks)
peaks = np.pad(peaks, (1,0), 'constant', constant_values=0)
print('Peaks: {}'.format(list(peak for peak in peaks)))


cap = cv2.VideoCapture(args.video)
total_frame=cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
print ("Total frame : {}, FPS : {}".format(total_frame, fps))

if args.display:
    current = 0
    now = 0
    while 1:
        ret, frame = cap.read()
        if not ret:
            print("END")
            break

        cv2.putText(frame, 'frame:{}, video:{}'.format(current, now), (10, 80), cv2.FONT_HERSHEY_PLAIN,
                    6, (0, 0, 255), 4, cv2.LINE_AA)
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

if args.store:

    pathOut = 'tiny'
    ret, frame = cap.read()
    if not ret:
        print('error')
        os._exit(0)
    size = (frame.shape[1]//4, frame.shape[0]//4)
    convert_frames_to_video(cap, pathOut, fps, peaks, size)
