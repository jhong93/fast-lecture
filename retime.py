#!/usr/bin/env python3

import os
import argparse
import math
import numpy as np
import whisper
from scipy.ndimage import maximum_filter1d, gaussian_filter1d
from moviepy.editor import VideoFileClip, VideoClip
from moviepy.audio.AudioClip import AudioArrayClip

from util.vtt import write_vtt, Segment

# Increase to make less agressive
MAX_SILENCE_MS = 75

PAD_WORD_MS = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file')
    parser.add_argument('-m', '--whisper_model', default='medium')
                        # large model has issues with word-timestamps?
    parser.add_argument('-o', '--out_file', required=True)
    return parser.parse_args()


def process_audio(vtt, audio, fade=5):
    fps = audio.fps
    arr = audio.to_soundarray()
    mask = np.zeros(arr.shape[0], dtype=bool)
    for l in vtt:
        start = int(l.start * fps)
        end = int(l.end * fps)
        mask[start:end] = True

    filter_width = int(fade * fps / 1000)
    weights = gaussian_filter1d(mask.astype(float), filter_width)
    new_arr = arr[mask, :] * weights[mask, None]
    return AudioArrayClip(new_arr, fps=audio.fps)


def process_mask_audio(mask, audio):
    arr = audio.to_soundarray()
    upsampled_mask = np.zeros(arr.shape[0], dtype=bool)
    num_frames = arr.shape[0]
    mask_len = mask.shape[0]
    for i in range(0, mask_len):
        if mask[i]:
            start = round(i / mask_len * num_frames)
            end = math.ceil((i + 1) / mask_len * num_frames)
            upsampled_mask[start:end] = True
    new_arr = arr[upsampled_mask, :]
    return AudioArrayClip(new_arr, fps=audio.fps)


def process_video(vtt, video):
    fps = video.fps
    num_frames = video.reader.nframes

    mask = np.zeros(num_frames, dtype=bool)
    for l in vtt:
        start = int(l.start * fps)
        end = int(l.end * fps)
        mask[start:end] = True
    idxs = np.where(mask)[0]

    pct = np.sum(mask) / mask.shape[0] * 100
    print(f'Has speech: {pct:0.3f}%')

    def make_frame(t):
        i = int(t * fps)
        i = min(i, idxs.shape[0] - 1)
        return video.get_frame(idxs[i] / fps)

    new_video = VideoClip(make_frame, duration=idxs.shape[0] / fps)
    return new_video, mask, fps


def process_subs(vtt, video, mask):
    fps = video.fps
    ofs = np.cumsum(mask)
    subs = []
    for l in vtt:
        start = int(l.start * fps)
        end = int(l.end * fps)
        if start == end:
            continue
        start_ofs = ofs[start] / fps
        end_ofs = ofs[end] / fps
        subs.append(Segment(start_ofs, end_ofs, l.text))
    return subs


def remove_silence(vtt, audio, threshold=0.1):
    arr = audio.to_soundarray()
    arr = np.mean(arr, axis=1)
    arr = np.abs(arr)
    window_frames = int(MAX_SILENCE_MS * audio.fps / 1000)

    arr_max = maximum_filter1d(arr, window_frames)
    assert arr_max.shape == arr.shape, (arr_max.shape, arr.shape)
    mask = arr_max > threshold
    pct = np.sum(mask) / mask.shape[0] * 100
    print(f'Has sound: {pct:0.3f}%')

    new_vtt = []
    for l in vtt:
        start = int(l.start * audio.fps)
        end = int(l.end * audio.fps)
        if end == start or np.any(mask[start:end]):
            if end != start:
                tmp = np.argwhere(mask[start:end]).flatten()
                l = l._replace(start=(start + tmp[0]) / audio.fps,
                               end=(start + tmp[-1]) / audio.fps)
            new_vtt.append(l)
    return new_vtt


def get_word_level_vtt(video_file, model):
    model = whisper.load_model(model)
    result = model.transcribe(video_file, verbose=True,
                              word_timestamps=True, language='en')

    lines = []
    for segment in result['segments']:
        words = segment['words']
        for w in words:
            start = w['start']
            end = w['end']
            text = w['word'].strip()
            lines.append(Segment(start, end, text))
    return lines


def main(args):
    video = VideoFileClip(args.video_file)
    print('Video:')
    print('  size:', video.size)
    print('  duration:', video.duration)
    print('  num frames:', video.reader.nframes)
    print('  fps:', video.fps)
    print()

    audio = video.audio
    print('Audio:')
    print('  nchannels:', audio.nchannels)
    print('  duration:', audio.duration)
    print('  num frames:', audio.reader.nframes)
    print('  fps:', audio.fps)
    print()

    vtt = get_word_level_vtt(args.video_file, args.whisper_model)

    # Quick sanity check
    for i in range(len(vtt)):
        assert vtt[i].end >= vtt[i].start
        if i > 0:
            assert vtt[i].start >= vtt[i - 1].start

    # Suppress halucinations from ASR in silent parts. Also trim silence.
    vtt = remove_silence(vtt, audio)

    if PAD_WORD_MS > 0:
        print('Padding (ms):', PAD_WORD_MS)
        vtt = [x._replace(start=max(0, x.start - PAD_WORD_MS / 1000),
                          end=x.end + PAD_WORD_MS / 1000) for x in vtt]

    # Resample the video
    new_video, mask, fps = process_video(vtt, video)

    # Resample the audio
    new_audio = process_mask_audio(mask, audio)
    print('New audio:')
    print('  nchannels:', new_audio.nchannels)
    print('  duration:', new_audio.duration)
    print()

    new_video = new_video.set_audio(new_audio)
    new_vtt = process_subs(vtt, video, mask)

    if args.out_file is not None:
        new_vtt_file = os.path.splitext(args.out_file)[0] + '.vtt'
        write_vtt(new_vtt_file, new_vtt)
        new_video.write_videofile(args.out_file, fps=fps,
                                  audio_codec='aac', codec='libx264')
    print('Done!')


if __name__ == '__main__':
    main(get_args())