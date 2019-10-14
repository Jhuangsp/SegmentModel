
import argparse
import time
import os,sys
import glob
import shutil
import numpy as np
import cv2
import imutils
import tensorflow as tf
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression
from imutils import paths

from cnn.Solver import Solver
from cnn.Resnet import Resnet
import DataProcess
import utils.utils as utils

parser = argparse.ArgumentParser(description='Skeleton-based action segment RNN model.', fromfile_prefix_chars='@')

# Data parameter
parser.add_argument('-i', '--images_path', type=str, default='data/demo',
                    help='Path to images dir (default:\"data/demo\")')
parser.add_argument('--openpose', type=str, default='/openpose-master',
                    help='Path to OpenPose dir (default:\"/openpose-master\")')
parser.add_argument('--output_dir', type=str, default='./openpose_out',
                    help='Path to Output dir (default:\"openpose_out\")')

parser.add_argument('-d', '--data_path', type=str, default='data',
                    help='Path to Dataset (default:\"data\")')
parser.add_argument('-jnum', '--num_joint', type=int, default=18,
                    help='Number of joints (default:18)')
parser.add_argument('-jdim', '--coord_dim', type=int, default=2,
                    help='Dimension of joint coordinate (default:2)')

# Display parameter
parser.add_argument('--test', action="store_true",
                    help='Only do inference')

# Model structure parameter
parser.add_argument('-ds', '--decoder_steps', type=int, default=3,
                    help='Steps of decoding (default:3)')
parser.add_argument('-if', '--in_frames', type=int, default=20,
                    help='Number of frames in one input sequence (default:20)')
parser.add_argument('-ob', '--out_band', type=int, default=10,
                    help='Number of output-band frames in the mid of input sequence (default:10)')
args = parser.parse_args()

def main():
    # Pre-load
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Offline
    '''images = glob.glob(args.images_path+'/*.png')
    for imagePath in images:
       image = cv2.imread(imagePath)'''
    # Realtime
    cap = cv2.VideoCapture('/dev/video0')
    
    checkpoint = "./model/best_model.ckpt" if True else "./model/trained_model.ckpt"
    loaded_graph = tf.Graph()

    # OpenPose
    openpose = '{}/examples/openpose/openpose.bin'.format(args.openpose)
    video_name = 'DEMO'
    output_snippets_dir = './data/openpose_estimation/snippets/{}'.format(video_name)
    output_sequence_dir = './data/openpose_estimation/data'
    output_sequence_path = '{}/{}.json'.format(output_sequence_dir, video_name)
    output_result_dir = args.output_dir
    output_result_path = '{}/{}.mp4'.format(output_result_dir, video_name)
    label_name_path = './label_name.txt'
    with open(label_name_path) as f:
        label_name = f.readlines()
        label_name = [line.rstrip() for line in label_name]
    openpose_args = dict(
        #video=self.args.video,
        image_dir=args.images_path,
        write_keypoint_json=output_snippets_dir,
        no_display='',
        render_pose=0, 
        model_pose='COCO')
    command_line = openpose + ' '
    command_line += ' '.join(['--{} {}'.format(k, v) for k, v in openpose_args.items()])
    shutil.rmtree(output_snippets_dir, ignore_errors=True)
    os.makedirs(output_snippets_dir)

    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)
        input_data = loaded_graph.get_tensor_by_name('resnet/X:0')
        logits = loaded_graph.get_tensor_by_name('train_result:0')

        # HOG estimation
        ret, image = cap.read()
        start_openpose = False
        start_threshold = 30 # frames
        start_count = 0
        buff_count = 0
        infer_threshold = 30
        buff = []
        answer_logits = []
        timer_count = 500

        print('123123123132132131212313213212313212123123132')
        print(ret)
        while ret:
            print('123123123132132131')
            timer_count -= 1
            if timer_count < 0:
                print('Time\'s up. waiting for result...')
                a_logits = oblique_mean(np.array(answer_logits))
                a_logits = np.pad(a_logits, (5, 5), 'edge')
                # peak detect
                print('\npeakdetect\n')
                # TODO

                start_openpose = False
                buff_count = 0
                buff = []
                answer_logits = []
                timer_count = 500
                time.sleep(5)

            if not start_openpose:
                print('Please sit down and stare at the camera.')
                h,w,_ = image.shape
                image = image[:(h*2)//3, w//3:(w*2)//3]
                faces = face_cascade.detectMultiScale(image, 1.1, 4)
                if len(faces) != 0:
                    start_count += 1
                else:
                    start_count = 0
                if start_count >= start_threshold:
                    start_count = 0
                    start_openpose = True
                    print('Start')
            else:
                print('Testing')

                filename = os.path.join(args.images_path, 'buff_{:08d}.png'.format(buff_count))
                buff.append(filename)
                cv2.imwrite(filename, image)
                buff_count += 1
                # end_count = 0
                if buff_count % infer_threshold == 0:
                    # OpenPose estimation
                    os.system(command_line)

                    # Load data
                    frames_list = glob.glob('data/openpose_estimation/snippets/DEMO/*.json')
                    num_frames = len(frames_list)
                    frame_data = np.zeros((num_frames, 18, 2), dtype=np.float32)
                    for i,frame in enumerate(frames_list):
                        with open(frame) as json_file:
                            frame_data[i] = utils.normalize(np.array(json.load(json_file)['people'][0]['pose_keypoints']), 'cnn')

                    for file in frames_list[:-10]:
                        os.remove(file)
                    for file in buff:
                        os.remove(file)
                    buff.clear()

                    # Segment model
                    # Build graph & Test
                    print('Run SegmentModel')
                    length = num_frames - (20-1)
                    indata = []
                    for i in range(length):
                        indata.append(frame_data[i:i+20, :])
                    answer_logits += list(sess.run(logits, { input_data: np.array([indata])}))

            cv2.waitKey(20)
            ret, image = cap.read()
        print('ENDENDENDENDENDENDENDENDENDENDENDENDENDEND')

if __name__ == '__main__':
    main()