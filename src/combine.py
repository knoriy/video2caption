import sys
import PIL
import json
import torch
import open_clip
from whisper_jax import FlaxWhisperPipline

import torchmetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datamodule import YoutubeTDM

whisperjax_model = FlaxWhisperPipline("openai/whisper-large-v2")
model, _, transform = open_clip.create_model_and_transforms(
    "coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)
model.to(device)


#############
# create global completed list
complete = []
#############




def inference_caption(image, decoding_method="Beam search", rep_penalty=1.2, top_p=0.5, min_seq_len=5, seq_len=20):
    im = transform(image).unsqueeze(0).to(device)
    generation_type = "beam_search" if decoding_method == "Beam search" else "top_p"
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(
            im, 
            generation_type=generation_type,
            top_p=float(top_p), 
            min_seq_len=min_seq_len, 
            seq_len=seq_len, 
            repetition_penalty=float(rep_penalty)
        )
    return open_clip.decode(generated[0].detach()).split("<end_of_text>")[0].replace("<start_of_text>", "")


def get_strategy(strategy:str='mse'):
    if strategy == 'mse':
        return torch.nn.MSELoss()
    elif strategy == 'ssim':
        return torchmetrics.StructuralSimilarityIndexMeasure()
    else:
        raise ValueError(f"Strategy {strategy} not supported")


def get_video_frames(frames, strategy:str='mse', threshold=0.5):
    strategy = get_strategy(strategy)

    frames = (frames/255).unsqueeze(1).permute(0,1,4,2,3)

    video_frames = []
    for i, frame in enumerate(frames):
        if not len(video_frames):
            video_frames.append((i, frame))

        difference = strategy(video_frames[-1][1], frame)

        if strategy == 'mse':
            difference = 1-difference

        if difference > threshold:
            video_frames.append((i, frame))

    return [(frame[0], (frame[1].squeeze(0).permute(1,2,0)*255).type(torch.uint8)) for frame in video_frames]

def main(urls:list[str], exclude_list:list[str]=[]):
    dataset = YoutubeTDM(
        train_urls=urls,
        exclude_list=exclude_list,
        )
    dataset.setup()

    for video in dataset.train:
        (video_frames, audio_frames, meta), json_meta = video

        if audio_frames.size(0) > 1:
            audio_frames = audio_frames[0]

        audio_sample = {"array": audio_frames.numpy(), 'sampling_rate':meta["audio_fps"]}

        text = whisperjax_model(audio_sample, return_timestamps=True)
        data = []
        for chunk in text["chunks"]:
            start_frame = int(chunk["timestamp"][0]*meta["video_fps"])
            end_frame = int(chunk["timestamp"][1]*meta["video_fps"])

            frames = get_video_frames(video_frames[start_frame:end_frame])
            breakpoint()

            captions = [{
                'frame':i+start_frame,
                "time": chunk["timestamp"][0] + (i/meta["video_fps"]), 
                'caption':inference_caption(PIL.Image.fromarray(frame.numpy()))} for (i, frame) in frames]
            
            data.append({"captions": captions, "gender": None, "emotion": None, "text": chunk['text'], "timestamp": chunk["timestamp"]})

        ###########
        # DO SOMETING WITH DATA
        ###########

        complete.append({"key": json_meta["key"], "id":json_meta["yt_meta_dict"]["info"]["id"]})
    return data


# Signal Handeling

def terminateProcess(signalNumber, frame):
    print ('(SIGTERM) terminating the process')

    # Cache completed files
    with open("completed.json", "w") as f:
        json.dump(complete, f)

    sys.exit()

if __name__ == '__main__':
    import signal
    import pprint

    # catch signal
    signal.signal(signal.SIGTERM, terminateProcess)

    pprint.pprint(main(["s3://s-laion/documentaries-videos/00000/"]))
    terminateProcess(None, None)