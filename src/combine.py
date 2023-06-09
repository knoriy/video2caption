import os
import io
import sys
import PIL
import json
import tqdm
import boto3
import torch
import open_clip
import tarfile

from functools import partial
from whisper_jax import FlaxWhisperPipline

import torchmetrics
import torchvision as tv
import torchaudio as ta

ta.backend.set_audio_backend("soundfile")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datamodule import YoutubeTDM

#############
# create global completed list
complete = []
#############

def inference_caption(model, transform, image, decoding_method="Beam search", rep_penalty=1.2, top_p=0.5, min_seq_len=5, seq_len=20):
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


def get_strategy(strategy:str='mse', device:str='cpu'):
    if strategy == 'mse':
        return torch.nn.MSELoss().to(device)
    elif strategy == 'ssim':
        return torchmetrics.StructuralSimilarityIndexMeasure().to(device)
    else:
        raise ValueError(f"Strategy {strategy} not supported")


def get_video_frames(frames, strategy, threshold=0.5):
    device = strategy.device
    frames = (frames/255).unsqueeze(1).permute(0,1,4,2,3)

    video_frames = []
    for i, frame in enumerate(frames):
        if i % 10 !=0:
            continue

        if not len(video_frames):
            video_frames.append((i, frame))

        difference = strategy(video_frames[-1][1].to(device), frame.to(device))

        if strategy == 'mse':
            difference = 1-difference

        if difference > threshold:
            video_frames.append((i, frame.detach().cpu()))

    return [(i, (frame.squeeze(0).permute(1,2,0)*255).type(torch.uint8)) for (i, frame) in video_frames]

def main(urls:list[str], exclude_list:list[str]=[], completed_path='completed.json', strategy='ssim'):
    dataset = YoutubeTDM(
        train_urls=urls,
        exclude_list=exclude_list,
        )
    dataset.setup()
    
    s3 = boto3.client('s3')
    bucket_name = 's-laion'

    whisperjax_model = FlaxWhisperPipline("openai/whisper-large-v2")
    model, _, transform = open_clip.create_model_and_transforms(
        "coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    model.to(device)

    strategy = get_strategy(strategy, device)

    for video in tqdm.tqdm(dataset.train, desc=f"Processing videos"):
        (video_frames, audio_frames, meta), json_meta = video
        del json_meta['yt_meta_dict']['subtitles']

        audio_frames = torch.mean(audio_frames, dim=0)
        audio_sample = {"array": audio_frames.numpy(), 'sampling_rate':meta["audio_fps"]}

        text = whisperjax_model(audio_sample, return_timestamps=True)

        ###########
        # Resample Audio to be saved
        ###########
        audio_frames = ta.transforms.Resample(meta["audio_fps"], 48000)(audio_frames.unsqueeze(0))
        meta["audio_fps"] = resamplerate = 48000

        print(json_meta["filename"])

        base_path, filename = json_meta["filename"].split("/")
        filename = os.path.join(base_path, f'{filename}')
        object_key = os.path.join('knoriy/documentaries-videos/', f"{filename}.tar")

        tar_butter = io.BytesIO()
        with tarfile.open(fileobj=tar_butter, mode='w') as tar:
            for chunk_index, chunk in enumerate(tqdm.tqdm(text["chunks"], desc=f"chunk")):
                try:
                    start_v_frame = int(chunk["timestamp"][0]*meta["video_fps"])
                    start_a_frame = int(chunk["timestamp"][0]*meta["audio_fps"])
                except:
                    start_v_frame = 0
                    start_a_frame = 0
                try:
                    end_v_frame = int(chunk["timestamp"][1]*meta["video_fps"])
                    end_a_frame = int(chunk["timestamp"][1]*meta["audio_fps"])
                except:
                    end_v_frame = -1
                    end_a_frame = -1

                start_chunk_timestamp = chunk["timestamp"][0]
                end_chunk_timestamp = chunk["timestamp"][1]
                if start_chunk_timestamp is None:
                    start_chunk_timestamp = 0
                if end_chunk_timestamp is None:
                    end_chunk_timestamp = float('inf')

                frames = get_video_frames(video_frames[start_v_frame:end_v_frame], strategy, threshold=0.9)

                buffer = io.BytesIO()
                ta.save(buffer, audio_frames[0][start_a_frame: end_a_frame].unsqueeze(0), resamplerate, format='flac')
                buffer.seek(0)
                audio_info = tarfile.TarInfo(f"{filename}_{chunk_index}.flac")
                audio_info.size = buffer.getbuffer().nbytes
                tar.addfile(audio_info, fileobj=buffer)

                captions = {}
                for frame_index, (i, frame) in enumerate(frames):
                    path = f"{filename}_{chunk_index}_{frame_index}.jpg"
                    captions[frame_index] = {"framepath":path ,'frame':i+start_v_frame, "time": start_chunk_timestamp + (i/meta["video_fps"]), 'caption':inference_caption(model, transform, PIL.Image.fromarray(frame.numpy()))}
                    buffer = io.BytesIO()
                    tv.utils.save_image(frame.permute(2,0,1)/255, buffer, format='jpeg')
                    buffer.seek(0)
                    info = tarfile.TarInfo(path)
                    info.size = buffer.getbuffer().nbytes
                    tar.addfile(info, fileobj=buffer)

                _data = {'filepath':os.path.join(*filename.split('/')[1:]), "captions": captions, "gender": None, "emotion": None, "text": chunk['text'], "timestamp": chunk["timestamp"], "original_data": json_meta}

                buffer = io.BytesIO()
                json_bytes = json.dumps(_data, ensure_ascii=False).encode('utf-8')
                buffer.write(json_bytes)
                buffer.seek(0)
                text_info = tarfile.TarInfo(f"{filename}_{chunk_index}.json")
                text_info.size = buffer.getbuffer().nbytes
                tar.addfile(text_info, fileobj=buffer)
        # Upload to S3
        s3.put_object(Bucket=bucket_name, Key=object_key, Body=tar_butter.getvalue())

        complete.append({"filename":json_meta["filename"] ,"key": json_meta["key"], "id":json_meta["yt_meta_dict"]["info"]["id"]})
        with open(completed_path, "w") as f:
            json.dump(complete, f)

    return complete


# Signal Handeling

def terminateProcess(signalNumber, frame, filename="completed.json"):
    print ('(SIGTERM) terminating the process')

    # Cache completed files
    with open(filename, "w") as f:
        json.dump(complete, f)

    sys.exit()

if __name__ == '__main__':
    import argparse
    import signal
    import pprint

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--urls', type=str, nargs='+', help='urls to process')
    args = parser.parse_args()

    completed_path = f"completed_{'_'.join([os.path.basename(url) for url in args.urls])}.json"
    print(completed_path)

    # catch signal
    signal.signal(signal.SIGTERM, partial(terminateProcess, filename=completed_path))
    signal.signal(signal.SIGUSR1, partial(terminateProcess, filename=completed_path))

    try:
        with open(completed_path, "r") as f:
            complete = json.load(f)
        exclude_list = [item["filename"] for item in complete]
    except FileNotFoundError:
        exclude_list = []

    pprint.pprint(exclude_list)

    pprint.pprint(main(args.urls, exclude_list=exclude_list, completed_path=completed_path))
    terminateProcess(None, None)