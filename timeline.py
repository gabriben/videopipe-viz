import subprocess
import pandas as pd
import numpy as np
import moviepy.editor as mp
import matplotlib.pyplot as plt
import seaborn as sns
from moviepy.video.io.bindings import mplfig_to_npimage
import core_viz as core


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


def read_shot_detection(v_name, task):
    '''
    Read the JSON file with the shot detection data.
    '''
    shots = pd.read_json(f"{v_name}/{v_name}{task}.json", lines=True)
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
    global fps

    if amount_of_frames_per_delay > 0:
        if delay_frames_left == 0:
            delay_frames_left = amount_of_frames_per_delay
        elif delay_frames_left < amount_of_frames_per_delay:
            delay_frames_left -= 1
            total_frames_delay += 1
        elif int(t * clip.fps - total_frames_delay) in delay_frames_set:
            delay_frames_left -= 1
            total_frames_delay += 1

        return t * clip.fps - total_frames_delay

    return t * clip.fps


if __name__ == '__main__':
    video_path = 'Videos/'
    v_name = 'HIGH_LIGHTS_I_SNOWMAGAZINE_I_SANDER_26'
    task = '_shot_boundaries_datamodel'
    detection_delay_sec = 1

    faces_detected = read_shot_detection(v_name, task)
    v_name = video_path + v_name

    clip = core.read_clip(v_name)
    fps = clip.fps
    data = [face['dimension_idx'] for face in faces_detected]

    fig, ax = make_timeline(clip,
                            data,
                            add_detection_indicator=True)

    # Global variables used to calculate the place of the time indicator
    # in the detection frequency plot animation.
    amount_of_frames_per_delay = detection_delay_sec * fps
    total_frames_delay = 0
    delay_frames_set = set(data)
    delay_frames_left = amount_of_frames_per_delay

    last_line = None

    def make_frame(t):
        global last_line

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
                             duration=total_video_time,)

    # TODO: debug timeline length too short.




