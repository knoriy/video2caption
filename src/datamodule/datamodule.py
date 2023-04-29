import os
import sys
import json
import torchdata

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from .base import BaseTDM
from utils import read_video

class YoutubeTDM(BaseTDM):
	def to_sampels(self, data):
		meta = data[0]
		video = data[2]
		audio = data[1]
		
		video_frames, _, video_meta = read_video(video[1], pts_unit="sec", end_pts=5)
		_, audio_frames, audio_meta = read_video(audio[1], pts_unit="sec", end_pts=5)
		json_meta = json.load(meta[1])

		return (video_frames, audio_frames, video_meta|audio_meta), json_meta

	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.list_files_by_fsspec()\
			.sharding_filter()\
			.open_files_by_fsspec(mode='rb')\
			.groupby(lambda x: os.path.basename(x[0]).split(".")[0], group_size=3, guaranteed_group_size=3)\
			.map(self.to_sampels) \

		return datapipe

	def collate_fn(self, batch):
		return batch