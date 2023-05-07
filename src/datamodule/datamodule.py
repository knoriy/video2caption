import os
import sys
import json
import torchdata

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from .base import BaseTDM
from utils import read_video

class YoutubeTDM(BaseTDM):
	def __init__(self, exclude_list=[], *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.exclude_list = exclude_list

	def filter_fn(self, data):
		file_name = os.path.join(*data[0][0].split("/")[-2:]).split(".")[0]
		if file_name in self.exclude_list:
			return False
		return True

	def to_sampels(self, data):
		meta = data[0]
		video = data[1]
		
		video_frames, audio_frames, av_meta = read_video(video[1], pts_unit="sec", end_pts=30)
		json_meta = json.load(meta[1])
		json_meta['filename'] = os.path.join(*video[0].split("/")[-2:]).split(".")[0]

		return (video_frames, audio_frames, av_meta), json_meta

	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.list_files_by_fsspec()\
			.sharding_filter()\
			.open_files_by_fsspec(mode='rb')\
			.groupby(lambda x: os.path.basename(x[0]).split(".")[0], group_size=2, guaranteed_group_size=2)\
			.filter(self.filter_fn)\
			.map(self.to_sampels) \

		return datapipe

	def collate_fn(self, batch):
		return batch