import cv2
import numpy as np
import matplotlib.pyplot as plt

def optical_flow():
    size = (224, 224)

    cap = cv2.VideoCapture("hmdb51/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi")
    ret, previous_frame = cap.read()

    if ret:
        # resize frame
        frame = cv2.resize(previous_frame, size)

        # convert to gray
        previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_frame = np.float32(previous_frame)/255.0

        # upload pre-processed frame to GPU
        gpu_previous = cv2.cuda_GpuMat(size, cv2.CV_32FC1)
        gpu_previous.upload(previous_frame)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, size)

            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_frame = np.float32(current_frame) / 255.0

            gpu_current = cv2.cuda_GpuMat(size, cv2.CV_32FC1)
            gpu_current.upload(current_frame)

            gpu_flow = cv2.cuda_BroxOpticalFlow.create(0.197, 50.0, 0.8, 5, 150, 10)
            gpu_flow = cv2.cuda_BroxOpticalFlow.calc(gpu_flow, gpu_previous, gpu_current, None,)

            optical_flow = gpu_flow.download()

            gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
            gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
            
            cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

            optical_flow_x = gpu_flow_x.download()
            optical_flow_y = gpu_flow_y.download()

            a = np.concatenate((optical_flow, np.zeros((224,224,1))), axis=2)

            dist_x = optical_flow_x.max() - optical_flow_x.min()
            dist_y = optical_flow_y.max() - optical_flow_x.min()

            gpu_previous.upload(current_frame)

if __name__ == '__main__':
    optical_flow()