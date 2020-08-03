import argparse
import time

import cv2
import threading

from sys import platform
from .models import *  # set ONNX_EXPORT in models.py
from .utils.datasets import *
from .utils.utils import *
import gc
import environ


class SlimYoloObjectDetection(object):
    def __init__(self):
        # queues of frames
        self.input_queues = {}
        self.output_queues = {}
        self.output_queues_risks = {}
        self.processed_frames = Queue(maxlen=100)
        self.non_processed_frames = Queue(maxlen=10)
        self.processed_frames_for_risks = Queue(maxlen=10)
        APPS_DIR = environ.Path("/home/luis/Documentos/psbposas/yoloservice/yoloservice")
        self.dirname = APPS_DIR
        self.cfg = os.path.join(APPS_DIR, 'slimyolov3/cfg/prune_0.5_0.5_0.7.cfg')  # cfg file path
        self.data = os.path.join(APPS_DIR, 'slimyolov3/VisDrone2019/drone.data')
        self.weights = os.path.join(APPS_DIR, 'slimyolov3/weights/prune_0.5_0.5_0.7_final.weights')  # path to weights file
        self.img_size = 608  # inference size (pixels)
        self.conf_thres = 0.3  # object confidence threshold
        self.nms_thres = 0.5  # iou threshold for non-maximum suppression
        self.half = True
        self.put_box = True
        self.is_streaming = True
        print(APPS_DIR)
        # start consumer
        thread = threading.Thread(target=self.start_detection, args=())
        #thread.daemon = True
        thread.start()

    def prep_image(self, img0):
        # img0 = cv2.flip(img0, 1)  # flip left-right

        # Padded resize
        img, *_ = letterbox(img0, new_shape=self.img_size)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(
            img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img, img0

    def start_detection(self, save_txt=False, save_img=False, stream_img=True):
        with torch.no_grad():
            # (320, 192) or (416, 256) or (608, 352) for (height, width)
            img_size = (320, 192) if ONNX_EXPORT else self.img_size
            weights, half = self.weights, self.half

            # Initialize
            device = torch_utils.select_device(force_cpu=ONNX_EXPORT)

            # Initialize model
            model = Darknet(self.cfg, img_size)

            # Load weights
            if str(self.weights).endswith('.pt'):  # pytorch format
                model.load_state_dict(torch.load(
                    weights, map_location=device)['model'])
            else:  # darknet format
                _ = load_darknet_weights(model, self.weights)

            # Fuse Conv2d + BatchNorm2d layers
            # model.fuse()
            # torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference

            # Eval mode
            model.to(device).eval()

            # Export mode
            if ONNX_EXPORT:
                img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
                torch.onnx.export(
                    model, img, 'weights/export.onnx', verbose=True)
                return

            # Half precision
            half = half and device.type != 'cpu'  # half precision only supported on CUDA
            if half:
                model.half()

            # Get classes and colors
            classes = load_classes(self.dirname(
                parse_data_cfg(self.data)['names']))
            colors = [[random.randint(0, 255) for _ in range(3)]
                      for _ in range(len(classes))]

            # Run inference
            t0 = time.time()

            while True:
                for client_uuid in list(self.input_queues):
                    client_input_queue = self.input_queues[client_uuid]
                    if self.is_streaming == False:
                        break
                    try:
                        im0, index = client_input_queue.pop()
                    except EmptyQueueException as eqe:
                        if self.is_streaming == False:
                            break
                    else:
                        if type(im0) == type(None):
                            self.is_streaming = False
                            model = None
                            gc.collect()
                            break  # ¿will break?
                        t = time.time()

                        # Prep img
                        img, im0 = self.prep_image(im0)

                        # Get detections
                        img = torch.from_numpy(img).unsqueeze(0).to(device)
                        pred, _ = model(img)
                        det = non_max_suppression(
                            pred.float(), self.conf_thres, self.nms_thres)[0]

                        s = '%gx%g ' % img.shape[2:]  # print string
                        detected_objects = []
                        if det is not None and len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(
                                img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            # for c in det[:, -1].unique():
                            #    n = (det[:, -1] == c).sum()  # detections per class
                            #    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                            # Write results
                            for *xyxy, conf, _, cls in det:
                                if save_txt:  # Write to file
                                    with open(save_path + '.txt', 'a') as file:
                                        file.write(('%g ' * 6 + '\n') %
                                                (*xyxy, cls, conf))

                                if self.put_box:  # Add bbox to image
                                    label = '%s %.2f' % (classes[int(cls)], conf)
                                    obj = {'label': str(classes[int(
                                        cls)]), 'conf': float(conf), 'box': [int(val) for val in xyxy]}
                                    detected_objects.append(obj)
                                    plot_one_box(xyxy, im0, label=label,
                                                color=colors[int(cls)])
                        self.output_queues[client_uuid].append(
                            {"image_np": im0, "index": index})  # add image to queue
                        cv2.imwrite("test.jpg", im0)
                        self.output_queues_risks[client_uuid].append(
                            {"image_np": im0, "index": index, "detected_objects": detected_objects})  # add image to queue
                        # print(im0.shape)
                        #print('%sDone. (%.3fs)' % (s, time.time() - t))
        gc.collect()
        self.non_processed_frames = None
        self.processed_frames = None
        self.processed_frames_for_risks = None
        torch.cuda.empty_cache()
        model = None
        print('Done. (%.3fs)' % (time.time() - t0))
        return

    def create_client_queues(self, client_uuid):
        self.input_queues[client_uuid] = Queue(maxlen=10)
        self.output_queues[client_uuid] = Queue(maxlen=10)
        self.output_queues_risks[client_uuid] = Queue(maxlen=10)
    
    def delete_client_queues(self, client_uuid):
        client_input_queue = self.input_queues.pop(client_uuid, None)
        client_output_queue = self.output_queues.pop(client_uuid, None)
        if client_input_queue == None and client_output_queue == None:
            return False
        else:
            del client_input_queue
            del client_output_queue
            return True

