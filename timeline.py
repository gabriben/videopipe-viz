import pandas as pd
import numpy as np
import moviepy.editor as mp
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from PIL import ImageDraw

import core_viz as core


def read_face_detection(v_name, task):
    '''
    Read the face detection JSON file.
    '''
    faces = pd.read_json(f"{v_name}/{v_name}{task}.json", lines=True)
    faces_detected = [f for f in faces.data[0] if len(f['faces']) > 0]
    return faces_detected


def make_timeline(clip, detected_output, size, output_fn, detail_modifier=0.5):
    w, h = size
    total_frames = int(clip.duration * clip.fps)
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches

    fig, ax = plt.subplots(figsize=(w*px, h*px))
    sns.set_style('whitegrid')
    sns.kdeplot(np.array(detected_output), clip=(0, total_frames), bw_method=0.10 * detail_modifier)
    ax.set_xlim(0, total_frames)
    axis_frames = range(0, total_frames, total_frames // 10)
    axis_timestamps = [core.frame_number_to_timestamp(fr, clip.fps, format='seconds') for fr in axis_frames]
    ax.set_xticks(axis_frames, axis_timestamps)
    ax.get_yaxis().set_visible(False)
    plt.savefig(output_fn)


if __name__ == '__main__':

    video_path = 'Videos/'
    v_name = 'HIGH_LIGHTS_I_SNOWMAGAZINE_I_SANDER_26'
    task = '_frame_face_detection_datamodel'

    w, h = 1920, 1080

    faces_detected = read_face_detection(v_name, task)
    v_name = video_path + v_name

    clip = core.read_clip(v_name)
    data = [face['dimension_idx'] for face in faces_detected]

    make_timeline(clip, data, (w, h/5), 'timeline_test.jpg')

