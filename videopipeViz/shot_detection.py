from typing import Optional
import pandas as pd
import numpy as np
import moviepy.editor as mp
from PIL import Image
from videopipeViz.midroll_marker import make_frame_line
import videopipeViz.core_viz as core


class ShotDetectionStrat(core.BurnInStrategy):
    def __init__(self,
                 add_timeline,
                 add_tl_indicators,
                 # add_tl_graph,
                 freeze_frame_sec):
        super().__init__(add_timeline,
                         add_tl_indicators,
                         # add_tl_graph,
                         freeze_frame_sec,)

    def read_json(self, json_path, v_name, json_postfix):
        '''
        Read the JSON file with the shot detection data.
        '''
        shots = pd.read_json(json_path + v_name + '/' +
                             v_name + json_postfix + '.json',
                             lines=True)
        shots_detected = [f for f in shots.data[0]]
        return shots_detected

    def get_clips(self,
                  clip: mp.VideoFileClip,
                  shots_detected: list,
                  shot_frame_duration: int = 3,
                  timestamp_offset: int = 0):
        '''
        Returns a list of clips with the shot boundaries and the shots
        themselves and the last timestamp of the clip in the list.
        '''
        clips = []
        frame_duration = 1 / clip.fps
        for shot in shots_detected:
            shot_boundary_frame_number = shot['dimension_idx']
            shotclip = self._create_shot_clip(clip,
                                              shot_boundary_frame_number,
                                              shot_frame_duration)
            timestamp = shot_boundary_frame_number * frame_duration

            subclip = clip.subclip(timestamp,
                                   timestamp
                                   + shot['duration'] * frame_duration
                                   + frame_duration)

            clips.append(shotclip)
            clips.append(subclip)

        return clips, timestamp

    def _create_shot_clip(self,
                          clip: mp.VideoFileClip,
                          shot_end_frame_number: int,
                          shot_frame_duration: int = 3) -> mp.ImageClip:
        '''
        Create a frameline indicating the precision of the shot boundary and
        make it into a shot_frame_duration long clip.
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


def shotDetection(json_path: str,
                  video_path: str,
                  v_name: str,
                  out_path: str,
                  json_postfix: str = '_shot_boundaries_datamodel',
                  add_timeline=True,
                  add_tl_indicators=True,
                  # add_tl_graph=True,
                  freeze_frame_sec: Optional[int] = 3,
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
        freeze_frame_sec (int or None): The amount of seconds the shot freeze
                                        frame is displayed. Defaults to 3.
        frames_per_round (int, optional): Sets the amount of detections per
        round, can be optimized for performance. Defaults to 100.
    """
    strat = ShotDetectionStrat(add_timeline,
                               add_tl_indicators,
                               # add_tl_graph,
                               freeze_frame_sec)
    core.burn_in_video(strat,
                       json_path,
                       video_path,
                       v_name,
                       out_path,
                       json_postfix,
                       frames_per_round=frames_per_round)
