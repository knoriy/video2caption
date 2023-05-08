import os
import io
import sys
import PIL
import json
import tqdm
import torch
import open_clip
import tarfile

from functools import partial
from whisper_jax import FlaxWhisperPipline

import torchmetrics
import torchvision as tv
import torchaudio as ta

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


def get_strategy(strategy:str='mse'):
    if strategy == 'mse':
        return torch.nn.MSELoss()
    elif strategy == 'ssim':
        return torchmetrics.StructuralSimilarityIndexMeasure()
    else:
        raise ValueError(f"Strategy {strategy} not supported")


def get_video_frames(frames, strategy, threshold=0.5):

    frames = (frames/255).unsqueeze(1).permute(0,1,4,2,3)

    video_frames = []
    for i, frame in enumerate(frames):
        if i % 10 !=0:
            continue

        if not len(video_frames):
            video_frames.append((i, frame))

        difference = strategy(video_frames[-1][1], frame)

        if strategy == 'mse':
            difference = 1-difference

        if difference > threshold:
            video_frames.append((i, frame))

    return [(i, (frame.squeeze(0).permute(1,2,0)*255).type(torch.uint8)) for (i, frame) in video_frames]

def main(urls:list[str], exclude_list:list[str]=[], strategy='ssim'):
    dataset = YoutubeTDM(
        train_urls=urls,
        exclude_list=exclude_list,
        )
    dataset.setup()


    whisperjax_model = FlaxWhisperPipline("openai/whisper-large-v2")
    model, _, transform = open_clip.create_model_and_transforms(
        "coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    model.to(device)

    strategy = get_strategy(strategy)

    for video in tqdm.tqdm(dataset.train, desc=f"Processing videos"):
        (video_frames, audio_frames, meta), json_meta = video
        del json_meta['yt_meta_dict']['subtitles']

        if audio_frames.size(0) > 1:
            audio_frames = ((audio_frames[0] + audio_frames[1]) / 2 )
        else:
            audio_frames = audio_frames[0]

        audio_sample = {"array": audio_frames.numpy(), 'sampling_rate':meta["audio_fps"]}

        text = whisperjax_model(audio_sample, return_timestamps=True)

        ###########
        # Resample Audio to be saved
        ###########
        meta["audio_fps"] = resamplerate = 48000
        audio_frames = ta.transforms.Resample(meta["audio_fps"], 48000)(audio_frames.unsqueeze(0))

        for chunk_index, chunk in enumerate(tqdm.tqdm(text["chunks"], desc=f"chunk")):
            start_v_frame = int(chunk["timestamp"][0]*meta["video_fps"])
            start_a_frame = int(chunk["timestamp"][0]*meta["audio_fps"])

            try:
                end_v_frame = int(chunk["timestamp"][1]*meta["video_fps"])
                end_a_frame = int(chunk["timestamp"][1]*meta["audio_fps"])
            except:
                end_v_frame = -1
                end_a_frame = -1

            frames = get_video_frames(video_frames[start_v_frame:end_v_frame], strategy, threshold=0.9)

            # save audio chunk

            base_path, filename = json_meta["filename"].split("/")
            os.makedirs(os.path.join('processes', base_path), exist_ok=True)

            filename = os.path.join(base_path, f'{filename}')
            with tarfile.open(os.path.join('processes', f"{filename}.tar"), mode='a') as tar:
                buffer = io.BytesIO()
                ta.save(buffer, audio_frames[0][start_a_frame: end_a_frame].unsqueeze(0), resamplerate, format='flac')
                buffer.seek(0)
                audio_info = tarfile.TarInfo(f"{filename}_{chunk_index}.flac")
                audio_info.size = buffer.getbuffer().nbytes
                tar.addfile(audio_info, fileobj=buffer)

                captions = {}
                for frame_index, (i, frame) in enumerate(frames):
                    path = f"{filename}_{chunk_index}_{frame_index}.jpg"
                    captions[frame_index] = {"framepath":path ,'frame':i+start_v_frame, "time": chunk["timestamp"][0] + (i/meta["video_fps"]), 'caption':inference_caption(model, transform, PIL.Image.fromarray(frame.numpy()))}
                    
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
        complete.append({"filename":json_meta["filename"] ,"key": json_meta["key"], "id":json_meta["yt_meta_dict"]["info"]["id"]})

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

    # catch signal
    signal.signal(signal.SIGTERM, partial(terminateProcess, filename=completed_path))
    signal.signal(signal.SIGUSR1, partial(terminateProcess, filename=completed_path))

    try:
        with open(completed_path, "r") as f:
            complete = json.load(f)
        exclude_list = [item["filename"] for item in complete]
    except FileNotFoundError:
        exclude_list = []


    pprint.pprint(main(args.urls, exclude_list=exclude_list))
    terminateProcess(None, None)