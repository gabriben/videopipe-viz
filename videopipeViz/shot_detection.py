import os
import pandas as pd
import numpy as np
import moviepy.editor as mp
from PIL import Image
from videopipeViz.midroll_marker import make_frame_line
import videopipeViz.core_viz as core
import videopipeViz.timeline as tl


def read_shot_detection(json_path, v_name, json_postfix):
    '''
    Read the JSON file with the shot detection data.
    '''
    shots = pd.read_json(json_path + v_name + '/' +
                         v_name + json_postfix + '.json',
                         lines=True)
    shots_detected = [f for f in shots.data[0]]
    return shots_detected


def create_shot_clip(clip, shot_end_frame_number, shot_frame_duration=1):
    '''
    Create frame of length shotclip_dur with txt in the center.
    '''
    _, h = clip.size
    shot_info_frame = Image.new('RGB', clip.size)
    frame_line = make_frame_line(clip,
                                 shot_end_frame_number,
                                 frame_type="Shot ends")
    frame_line.thumbnail(clip.size, Image.LANCZOS)
    shot_info_frame.paste(frame_line, (0, int(h/2)))
    shotclip = mp.ImageClip(np.asarray(shot_info_frame),
                            duration=shot_frame_duration)

    return shotclip


def get_shot_clips(clip,
                   shots_detected,
                   shot_frame_duration=3,
                   timestamp_offset=0):
    '''
    Returns a list of clips with the shot boundaries and the shots themselves.
    shots_limit determines how many shot boundaries are included.
    '''
    clips = []
    frame_duration = 1 / clip.fps
    for shot in shots_detected:
        shot_boundary_frame_number = shot['dimension_idx']
        shotclip = create_shot_clip(clip,
                                    shot_boundary_frame_number,
                                    shot_frame_duration=shot_frame_duration)
        timestamp = shot_boundary_frame_number * frame_duration

        subclip = clip.subclip(timestamp,
                               timestamp
                               + shot['duration'] * frame_duration
                               + frame_duration)

        clips.append(shotclip)
        clips.append(subclip)

    return clips, timestamp


def shotDetection(json_path: str,
                  video_path: str,
                  v_name: str,
                  out_path: str,
                  json_postfix: str = '_shot_boundaries_datamodel',
                  add_timeline=True,
                  add_tl_indicators=True,
                  add_tl_graph=True,
                  frames_per_round: int = 100) -> None:
    """ Burns in the shot detection JSON in the video and
    adds a timeline animation on the bottom displaying the detection density.

    Args:
        json_path (str): path of the JSON folder
        video_path (str): path of the folder of the video.
        v_name (str): name of the original video.
        out_path (str): folder for the output video.
        json_postfix (str, optional): postfix of the JSON file.
                                      Defaults to '_shot_detection_datamodel'.
        add_timeline (bool, optional): Flag for the addition of the timeline.
                                       Defaults to True.
        add_tl_indicators (bool, optional): Flag for the timeline indicators.
                                            Defaults to True.
        add_tl_graph (bool, optional): Flag for the addition of the graphline.
                                       Currently UNUSED. Defaults to True.
        frames_per_round (int, optional): Sets the amount of detections per
        round, can be optimized for performance. Defaults to 100.
    """
    shot_detected = read_shot_detection(json_path, v_name, json_postfix)
    clip = core.read_clip(video_path + v_name)

    shot_frame_duration = 3
    total_rounds = len(shot_detected) // frames_per_round

    prev_ts = 0
    with open(out_path + 'shot_detection.txt', 'w') as f:
        for round in range(total_rounds + 1):
            clips = []
            start_shot_number = round * frames_per_round
            end_shot_number = start_shot_number + frames_per_round
            shot_batch = shot_detected[start_shot_number:end_shot_number]
            clips, prev_ts = get_shot_clips(clip,
                                            shot_batch,
                                            shot_frame_duration,
                                            prev_ts)

            if (round == total_rounds):
                clips.append(clip.subclip(prev_ts, clip.duration))

            core.write_clip(mp.concatenate_videoclips(clips),
                            v_name,
                            postfix=str(round),
                            audio=False)
            f.write('file ' + v_name + '_' + str(round) + '.mp4\n')

    new_v_name = out_path + v_name + json_postfix

    core.files_to_video(clip,
                        v_name,
                        total_rounds,
                        out_path + 'shot_detection.txt',
                        new_v_name + '.mp4')

    if not add_timeline:
        return

    shot_frame_numbers = [shot['dimension_idx'] for shot in shot_detected]
    timeline = tl.TimelineAnimation(clip,
                                    v_name,
                                    shot_frame_numbers,
                                    add_graph=add_tl_graph,
                                    add_indicators=add_tl_indicators,
                                    detection_delay_sec=0,
                                    task=json_postfix)

    timeline.add_to_video(new_v_name + '.mp4', new_v_name + '_timeline.mp4')

    if os.path.exists(new_v_name + '.mp4'):
        os.remove(new_v_name + '.mp4')
