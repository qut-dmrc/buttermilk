import cv2
from moviepy.editor import VideoFileClip
import time
import base64
from pathlib import Path
from pydantic import validate_call

@validate_call
def extract_video_frames_b64(video_path: Path, seconds_per_frame: float = 2):
    """ Extract frames from video for use in analysis."""
    
    base64Frames = []

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # # Extract audio from video
    # audio_path = video_path.with_suffix(".mp3")
    # clip = VideoFileClip(video_path)
    # clip.audio.write_audiofile(audio_path, bitrate="32k")
    # clip.audio.close()
    # clip.close()

    return base64Frames 
