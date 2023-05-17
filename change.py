import cv2
import os

# 修改视频分辨率的函数
def resize_video(video_path, save_path, width, height):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    success, frame = videoCapture.read()
    while success:
        frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_CUBIC)
        videoWriter.write(frame)
        success, frame = videoCapture.read()
    videoCapture.release()
    videoWriter.release()

if __name__ == '__main__':
    video_path = 'data/pubg.mp4'
    save_path = 'data/pubg_320x180.mp4'
    resize_video(video_path, save_path, 320, 180)