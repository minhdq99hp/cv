import os
import sys
import cv2
import argparse
import torch
from PIL import ImageGrab
from PIL import Image
import numpy as np
import time
import cv2

from detector import Detector

# cap = cv2.VideoCapture("./out.mp4")
# while not cap.isOpened():
#     cap = cv2.VideoCapture("./out.mp4")
#     cv2.waitKey(1000)
#     print "Wait for the header"

# pos_frame = cap.get(cv2.CV_CAP_PROP_POS_FRAMES)
# while True:
#     flag, frame = cap.read()
#     if flag:
#         # The frame is ready and already captured
#         cv2.imshow('video', frame)
#         pos_frame = cap.get(cv2.CV_CAP_PROP_POS_FRAMES)
#         print str(pos_frame)+" frames"
#     else:
#         # The next frame is not ready, so we try to read it again
#         cap.set(cv2.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
#         print "frame is not ready"
#         # It is better to wait for a while for the next frame to be ready
#         cv2.waitKey(1000)

#     if cv2.waitKey(10) == 27:
#         break
#     if cap.get(cv2.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.CV_CAP_PROP_FRAME_COUNT):
#         # If the number of captured frames is equal to the total number of frames,
#         # we stop
#         break

# os.environ["CUDA_VISIBLE_DEVICES"]=""


def main(args):
    if args.mode not in ('camera', 'image', 'video'):
        print("Invalid inferencing mode")
        sys.exit(1)
    
    if args.model not in ('yolo5m', 'yolo5s'):
        print("Unsupported model")
        sys.exit(1)
    
    if args.mode in ('image', 'video'):
        if not args.input or not os.path.exists(args.input):
            print("Input path doesn't exist")
            sys.exit(1)
        
        if not args.output:
            print("Must specify output path")
            sys.exit(1)
    
    print('Device: ', args.device, torch.cuda.is_available())
    
    detector = Detector(model_name=args.model, device=args.device, confidence=args.min_confidence, iou=args.min_iou)

    if args.mode == 'image':
        img = Image.open(args.input)
        res = detector.detect(np.asarray(img))

        res = Image.fromarray(res)
        res.save(args.output)
    elif args.mode == 'video':
        cap = cv2.VideoCapture(args.input)

        # pos_frame = cap.get(cv2.CV_CAP_PROP_POS_FRAMES)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width,frame_height))

        i = 0
        while True:
            i += 1
            flag, frame = cap.read()
            if flag:
                
                # The frame is ready and already captured
                # cv2.imshow('video', frame)
                # pos_frame = cap.get(cv2.CV_CAP_PROP_POS_FRAMES)
                # print str(pos_frame)+" frames"

                res = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # print(type(res))

                out.write(cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
                print(f"{i}/{total_frames}")
            else:
                # The next frame is not ready, so we try to read it again
                # cap.set(cv2.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
                # print "frame is not ready"
                # It is better to wait for a while for the next frame to be ready
                # cv2.waitKey(1000)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # if cv2.waitKey(10) == 27:
            #     break
            # if cap.get(cv2.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.CV_CAP_PROP_FRAME_COUNT):
            #     # If the number of captured frames is equal to the total number of frames,
            #     # we stop
            #     break
        
    elif args.mode == 'camera':
        cap = cv2.VideoCapture(0)

        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        resized_dim = (640, int(frame_height * 640 / frame_width))
        

        i = 0
        while True:
            i += 1
            flag, frame = cap.read()
            if flag:
                frame = cv2.resize(frame, resized_dim)

                res = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                cv2.imshow('output', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
                
                # print(f"{i}/{total_frames}")
            else:
                print('frame is not valid')
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cv2.destroyAllWindows()

    else:
        pass

    print('Done !')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference script for computer vision project')
    parser.add_argument('--mode', type=str, default='camera', help='choose mode for inferencing')
    parser.add_argument('--model', type=str, default='yolo5s', help='choose model for inferencing')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--min_confidence', type=float, default=0.5)
    parser.add_argument('--min_iou', type=int, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('integers', metavar='int', nargs='+', type=int, help='an integer to be summed')
    # parser.add_argument('--log', default=sys.stdout, type=argparse.FileType('w'), help='the file where the sum should be written')
    args = parser.parse_args()
    # args.log.write('%s' % sum(args.integers))
    # args.log.close()

    main(args)
