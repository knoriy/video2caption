import torchvision
from .video import read_video


def get_video_data(filename):
    video_frames, audio_frames, meta = torchvision.io.read_video(filename, pts_unit="sec")
    if audio_frames.shape[0] > 1:
        audio_frames = audio_frames[0]
    return video_frames, audio_frames, meta