import grpc, cv2, io, time
from PIL import Image
from concurrent import futures
from yoloservice.slimyolov3.slimyolo import SlimYoloObjectDetection
from yoloservice.generated import detectionservice_pb2_grpc, detectionservice_pb2

im = Image.open("cars.jpg")
buf = io.BytesIO()
im.save(buf, format="JPEG")
def spam_images():
    for i in range(100):
        time.sleep(0.2)
        yield detectionservice_pb2.Frame(request_img=buf.getvalue(), width=im.width, height=im.height ,frame_id=int(i))

channel = grpc.insecure_channel("localhost:5555")
stub = detectionservice_pb2_grpc.ProcessFramesStub(channel)
responses = stub.Process(spam_images())
image = None
for response in responses:
    img = response.response_img
    image = Image.open(io.BytesIO(img))
    image.show()
    break



