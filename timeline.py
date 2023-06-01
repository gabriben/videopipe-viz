import pandas as pd
import numpy as np
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


def make_timeline(clip, detected_output, size, detail_modifier=0.5):
    w, h = size
    total_frames = int(clip.duration * clip.fps)
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches

    fig, ax = plt.subplots(figsize=(w*px, h*px))
    sns.set_style('whitegrid')
    sns.kdeplot(np.array(detected_output),
                clip=(0, total_frames),
                bw_method=0.10 * detail_modifier)
    ax.set_xlim(0, total_frames)
    axis_frames = range(0, total_frames, total_frames // 10)
    axis_timestamps = [core.frame_number_to_timestamp(fr,
                                                      clip.fps,
                                                      format='seconds')
                       for fr in axis_frames]
    ax.set_xticks(axis_frames, axis_timestamps)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    video_path = 'Videos/'
    v_name = 'HIGH_LIGHTS_I_SNOWMAGAZINE_I_SANDER_26'
    task = '_frame_face_detection_datamodel'

    w, h = 1920, 1080

    faces_detected = read_face_detection(v_name, task)
    v_name = video_path + v_name

    clip = core.read_clip(v_name)
    fps = clip.fps
    data = [face['dimension_idx'] for face in faces_detected]

    fig, ax = make_timeline(clip,
                            data,
                            (w, h/7),
                            detail_modifier=0.3)

    plt.show()
    last_line = None

    def make_frame(t):
        global last_line
        global fps
        if last_line is not None:
            last_line.remove()
        last_line = ax.axvline(t * clip.fps,
                               color=(0.8, 0.3, 0.2),
                               linestyle='dashed',
                               linewidth=1)
        return mplfig_to_npimage(fig)

    # animation = mp.VideoClip(make_frame, duration=clip.duration)
    # core.write_clip(animation, "timeline_test", audio=False)


