from yoloservice import server
import grpc, threading
from yoloservice.generated import detectionservice_pb2, detectionservice_pb2_grpc

def run():
    server.serve()

thread_server = threading.Thread(target=run)
thread_server.start()