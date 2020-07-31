# Start from Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:19.08-py3

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Install dependencies (pip or conda)
# RUN pip install -U -r requirements.txt
# RUN conda update -n base -c defaults conda
# RUN conda install -y -c anaconda future numpy opencv matplotlib tqdm pillow
# RUN conda install -y -c conda-forge scikit-image tensorboard pycocotools
# conda install pytorch torchvision -c pytorch

# Move model into container
# RUN mv yolov3-spp.pt ./weights


# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build container
# rm -rf yolov3  # Warning: remove existing
# git clone https://github.com/ultralytics/yolov3 && cd yolov3 && python3 detect.py
# sudo docker image prune -af && sudo docker build -t ultralytics/yolov3:v0 .

# Run container
# time sudo nvidia-docker run ultralytics/yolov3:v0 python3 detect.py

# Push container to https://hub.docker.com/u/ultralytics
# sudo docker push ultralytics/xview:v30
