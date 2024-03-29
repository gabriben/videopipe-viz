import os
import subprocess
import pandas as pd
import numpy as np
import moviepy.editor as mp
import matplotlib.pyplot as plt
import seaborn as sns
from moviepy.video.io.bindings import mplfig_to_npimage
import videopipeViz.core_viz as core

import matplotlib
from sys import platform
if platform == "darwin":
    matplotlib.use("Qt5Agg")


def read_face_detection(v_name, task):
    '''
    Read the face detection JSON file.
    '''
    faces = pd.read_json(f"{v_name}/{v_name}{task}.json", lines=True)
    faces_detected = [f for f in faces.data[0] if len(f['faces']) > 0]
    return faces_detected


def read_text_detection(v_name, task):
    '''
    Read the text detection JSON file.
    '''
    text = pd.read_json(f"Videos/{v_name}/{v_name}{task}.json", lines=True)
    texts_detected = [f for f in text.data[0] if len(f['text']) > 0]
    return texts_detected


def read_shot_detection(json_path, v_name, task):
    '''
    Read the JSON file with the shot detection data.
    '''
    shots = pd.read_json(f"{json_path}{v_name}/{v_name}{task}.json", lines=True)
    shots_detected = [f for f in shots.data[0]]
    return shots_detected


def make_timeline(clip, detected_output, height_ratio=7,
                  add_detection_indicator=False, detail_modifier=1.0):
    DETAIL_SWEETSPOT = 0.03

    w, h = clip.size
    total_frames = int(clip.duration * clip.fps)
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches

    fig, ax = plt.subplots(figsize=(w * px, h / height_ratio * px))

    sns.set_style('whitegrid')
    g = sns.kdeplot(np.array(detected_output),
                    clip=(0, total_frames),
                    bw_method=DETAIL_SWEETSPOT * detail_modifier,
                    color='navy',
                    zorder=100)

    if add_detection_indicator:
        ymin, ymax = g.get_ylim()
        g.vlines(detected_output,
                 ymin=ymin,
                 ymax=ymax,
                 colors='lightblue',
                 lw=1,
                 zorder=0)

    midroll_indicator = 10664
    ax.axvline(midroll_indicator,
               color='orange',
               linestyle='solid',
               linewidth=2)

    ax.set_xlim(0, total_frames)
    # TODO: Optimize ticks
    axis_frames = range(0, total_frames, total_frames // 10)
    axis_timestamps = [core.frame_number_to_timestamp(fr,
                                                      clip.fps,
                                                      format='seconds')
                       for fr in axis_frames]
    
    ax.set_xticks(axis_frames, axis_timestamps)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    return fig, ax


def add_timeline_to_video(video_path, timeline_vid_path, output_filename):
    cmd = f"ffmpeg -i {video_path} -i {timeline_vid_path} -filter_complex vstack {output_filename}"
    subprocess.call(cmd, shell=True)


def calculate_time_indicator_frame_number(t):


    global amount_of_frames_per_delay
    global total_frames_delay
    global delay_frames_left
    global delay_frames_set
    

    # nonlocal delay_frames_left

    # nonlocal amount_of_frames_per_delay
    # nonlocal total_frames_delay
    # nonlocal delay_frames_left
    # nonlocal delay_frames_set
    # nonlocal fps
    

    current_frame = int(round(t * fps))
    if amount_of_frames_per_delay > 0:
        if delay_frames_left == 0:
            delay_frames_left = amount_of_frames_per_delay
        elif delay_frames_left < amount_of_frames_per_delay:
            delay_frames_left -= 1
            total_frames_delay += 1
        elif current_frame - total_frames_delay in delay_frames_set:
            delay_frames_set.discard(current_frame - total_frames_delay)
            delay_frames_left -= 1
            total_frames_delay += 1

        # +1 for the just added total_frames_delay
        return current_frame - total_frames_delay + 1

    print(t)
    
    return current_frame

def timeline(json_path: str,
             video_path: str,
             v_name: str,
             out_path: str,
             detection_delay_sec = 2) -> None:

    global fps
    global amount_of_frames_per_delay
    global total_frames_delay
    global delay_frames_left
    global delay_frames_set    
    
    task = '_shot_boundaries_datamodel'
    

    data_detected = read_shot_detection(json_path, v_name, task)
    
    clip = core.read_clip(video_path + v_name)
    fps = clip.fps
    data = [d['dimension_idx'] for d in data_detected]

    fig, ax = make_timeline(clip,
                            data,
                            add_detection_indicator=True)


    # Global variables used to calculate the place of the time indicator
    # in the detection frequency plot animation.
    amount_of_frames_per_delay = int(detection_delay_sec * fps)
    total_frames_delay = 0
    delay_frames_set = set(data)  #set([d for d in data])
    delay_frames_left = int(detection_delay_sec * fps)

    # import pdb
    # pdb.set_trace()

    last_line = None

    def make_frame(t):
        nonlocal last_line

        if last_line is not None:
            last_line.remove()

        time_indicator_frame_number = calculate_time_indicator_frame_number(t)
        last_line = ax.axvline(time_indicator_frame_number,
                               color=(1, 0, 0),
                               linestyle='dashed',
                               linewidth=1)

        return mplfig_to_npimage(fig)


    total_video_time = clip.duration + len(data) * detection_delay_sec

    animation = mp.VideoClip(make_frame,
                             duration=total_video_time)

    timeline_mp4 = out_path + v_name + task
    timeline_mp4_with_postfix = timeline_mp4 + '_midroll_delay_' + str(detection_delay_sec) + '.mp4'

    if not os.path.exists(timeline_mp4_with_postfix):
        core.write_clip(animation, timeline_mp4, postfix= 'midroll_delay_' + str(detection_delay_sec),
                    audio=False, logger = 'bar')

    add_timeline_to_video(video_path + v_name + ".mp4",
                          timeline_mp4_with_postfix,
                          out_path + v_name + task + "_timeline_midroll_delay" + str(detection_delay_sec) + ".mp4")




