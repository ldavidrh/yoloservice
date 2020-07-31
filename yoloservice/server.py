import grpc, io, random
import numpy as np
from PIL import Image
from concurrent import futures
from yoloservice.slimyolov3.slimyolo import SlimYoloObjectDetection
from yoloservice.generated import detectionservice_pb2_grpc, detectionservice_pb2

#yolo = SlimYoloObjectDetection()

class ReceiveFrame(detectionservice_pb2_grpc.ProcessFramesServicer):
    def Process(self, requests, context):
        frames = requests
        for frame in frames:
            img = Image.open(io.BytesIO(frame.request_img))
            imgarray = np.asarray(img)
            yield detectionservice_pb2.Detection(response_img=b'x9239x89x', detections='test_detections', frame_id=int(random.random()*10))
        
        

def serve():
    port = 5555
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    detectionservice_pb2_grpc.add_ProcessFramesServicer_to_server(ReceiveFrame(), server)
    server.add_insecure_port("[::]:%i"%port)
    server.start()
    print("Server running on port %s"% port)
    server.wait_for_termination()
