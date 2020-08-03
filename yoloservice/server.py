import grpc, io, random, json, cv2, uuid
import numpy as np
from PIL import Image
from concurrent import futures
from yoloservice.slimyolov3.slimyolo import SlimYoloObjectDetection
from yoloservice.generated import detectionservice_pb2_grpc, detectionservice_pb2


yolo = SlimYoloObjectDetection()
queues = {}

class ReceiveFrame(detectionservice_pb2_grpc.ProcessFramesServicer):
    def Process(self, requests, context):
        frames = requests
        for frame in frames:
            img = Image.open(io.BytesIO(frame.request_img))
            imgarray = np.asarray(img) 
            yolo.non_processed_frames.append((imgarray, int(frame.frame_id)))
            if yolo.processed_frames_for_risks.head != None:
                count = count + 1 
                processed_frame = yolo.processed_frames_for_risks.pop()
                detections = processed_frame["detected_objects"]
                detections = json.dumps(detections)
                index = processed_frame["index"]
                imgbytes = processed_frame["image_np"]
                imgbytes = imgbytes[:,:,[2, 1, 0]]
                success, imgbytes = cv2.imencode('.jpg', imgbytes)
                yield detectionservice_pb2.Detection(response_img=imgbytes.tostring(), detections=detections, frame_id=int(index))
        
        

def serve():
    port = 5555
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    detectionservice_pb2_grpc.add_ProcessFramesServicer_to_server(ReceiveFrame(), server)
    server.add_insecure_port("[::]:%i"%port)
    server.start()
    print("Server running on port %s"% port)
    server.wait_for_termination()

