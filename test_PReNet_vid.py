import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import time

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="logs/200709/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="/media/r/BC580A85580A3F20/dataset/rain/peku/Rain100H/rainy", help='path to training data')
parser.add_argument("--save_path", type=str, default="/home/r/works/derain_arxiv/release/results/PReNet", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--video_path", type=str, default="rtsp://172.26.19.202:554/h264", help="path to video file")
parser.add_argument("--rtsp_path", type=str, default="rtsp://172.26.19.202:554/h264", help="path to rtsp streaming link")
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

from threading import Thread

class RTSPVideoWriterObject(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc('M','J','P','G')
        self.output_video = cv2.VideoWriter('output.avi', self.codec, 30, (self.frame_width, self.frame_height))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        # Display frames in main program
        if self.status:
            cv2.imshow('frame', self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            self.output_video.release()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self):
        # Save obtained frame into video output file
        self.output_video.write(self.frame)

def main():

    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = PReNet(opt.recurrent_iter, opt.use_GPU)
    #print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_epoch3.pth')))
    model.eval()

    time_test = 0
    count = 0
    print("Pass")

    # Text Parmas
    green = (0, 255, 0)
    thickness = 2
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX  # hand-writing style font
    fontScale = 1
    location_origin = (50, 50)
    location_deweather = (50, 50)

    # Video Input
    cap = cv2.VideoCapture()
    #cap.open("rtsp://172.26.19.202:554/h264")
    #cap.open(opt.video_path)
    #cap.open("G:\\0000003.mp4")
    #cap.open("I:\\(15min)100-120dB.mp4")
    cap.open("./test_001.avi")

    print("Pass2")

    '''
    RTSP Part
    '''
    rtsp_stream_link = opt.rtsp_path  #'your stream link!'
    video_stream_widget = RTSPVideoWriterObject(rtsp_stream_link)

    gt = cv2.imread('./datasets/train/200717/ground_truth/frame0.jpg')
    gt = cv2.resize(gt, (640, 480), interpolation=cv2.INTER_AREA)

    while(cap.isOpened()):
        try:
            # input image
            ret, frame = cap.read()
            print("Read")
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            b, g, r = cv2.split(frame)
            y = cv2.merge([r, g, b])

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y))

            if opt.use_GPU:
                y = y.cuda()

            with torch.no_grad():  #
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()

                out, _ = model(y)
                out = torch.clamp(out, 0., 1.)

                if opt.use_GPU:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(count, ': ', dur_time)

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())  # back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            print(np.shape(save_out))
            cv2.imwrite(opt.save_path + 'SINGLE/%d.png' %count, save_out)
            print("Save!!")

            # Put Text on Video
            cv2.putText(frame, 'Origin', location_origin, font, fontScale, green, thickness)
            cv2.putText(save_out, 'De-Weather', location_deweather, font, fontScale, green, thickness)
            cv2.putText(gt, 'Ground-Truth', location_deweather, font, fontScale, green, thickness)

            # Multi-Window
            numpy_horizontal_concat = np.concatenate((frame, save_out, gt), axis=1)

            '''
            If you want to see only one window image,
            Change parameters 'numpy_horizontal_concat' to 'you want' in line 122

            frame : Original Video
            save_out : De-Weather Video

            - For example (see in line 122_
            1) For original video only
             > cv2.imshow("img1", frame)

            2) For de-weather video only
             > cv2.imshow("img1", save_out)
            '''

            cv2.imshow("img1", numpy_horizontal_concat)
            cv2.imwrite(opt.save_path + '/%d aa.png' % count, numpy_horizontal_concat)

            # rtsp로 다시 출력
            # L102줄을 참고하면 됨.
            #video_stream_widget.show_frame()
            #video_stream_widget.save_frame()

            cv2.waitKey(27)

            count += 1

        except Exception as e:
            print(str(e))
            pass



    print('Avg. time:', time_test/count)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

