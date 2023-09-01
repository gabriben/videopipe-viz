from typing import Optional, Tuple
import pandas as pd
import moviepy.editor as mp
from PIL import ImageDraw, Image, ImageFont
import numpy as np
import videopipeViz.core_viz as core


class TextDetectionStrat(core.BurnInStrategy):
    def __init__(self,
                 add_timeline,
                 add_tl_indicators,
                 # add_tl_graph,
                 freeze_frame_sec):
        super().__init__(add_timeline,
                         add_tl_indicators,
                         # add_tl_graph,
                         freeze_frame_sec)

    def read_json(self, json_path: str, v_name: str, json_postfix: str):
        '''
        Read the text detection JSON file.
        '''
        text = pd.read_json(json_path + v_name + '/'
                            + v_name + json_postfix + '.json',
                            lines=True)
        texts_detected = [f for f in text.data[0] if len(f['text']) > 0]
        return texts_detected

    def get_clips(self,
                  clip: mp.VideoFileClip,
                  texts_detected: list,
                  txt_frame_duration: int,
                  timestamp_offset: int = 0) -> Tuple[list, float]:
        """ Make a list of clips with all the text frames in 'texts_detected'.
        Text frames are inserted with a duration of 'txt_frame_duration'.
        'timestamp_offset' is used to determine the starting time of the first
        (textless) subclip.
        """
        clips = []
        for txt in texts_detected:
            ts, bb_frame = self._make_frame(clip, txt)

            if (timestamp_offset != ts):
                clips.append(clip.subclip(timestamp_offset, ts))

            txt_frame_clip = mp.ImageClip(np.asarray(bb_frame),
                                          duration=txt_frame_duration)
            clips.append(txt_frame_clip)
            timestamp_offset = ts + txt_frame_duration

        return clips, timestamp_offset

    def _draw_text_bb(self,
                      frame: Image.Image,
                      texts: dict,
                      color: str = 'blue',
                      bb_width: int = 5,
                      txtcolor: str = 'black') -> Image.Image:
        """ Draw all the detected text in 'texts' on top of the frame. """
        copy = frame.copy()
        font = ImageFont.truetype("NotoSansMono-Bold.ttf", 20)
        for txt in texts:
            left, top, width, height, conf, detected_text = texts[txt].values()
            right = left + width
            bottom = top + height
            draw = ImageDraw.Draw(copy)
            draw.rectangle((left, top, right, bottom),
                           outline=color,
                           width=bb_width)
            draw.text((left, bottom),
                      detected_text + "(" + str(conf) + ")",
                      font=font,
                      fill=txtcolor)
        return copy

    def _make_frame(self,
                    clip: mp.VideoFileClip,
                    txts: dict) -> Tuple[float, Image.Image]:
        """
        Get the frame in 'txts' and draw all the texts in 'txts' on the frame.
        Also return the timestamp in the clip of the detected frame.
        """
        txt_frame_number = txts['dimension_idx']
        txt_timestamp = txt_frame_number / clip.fps
        frame = core.get_frame_by_number(clip, txt_frame_number)
        bb_frame = self._draw_text_bb(frame, txts['text'])

        return txt_timestamp, bb_frame


def textDetection(json_path: str,
                  video_path: str,
                  v_name: str,
                  out_path: str,
                  json_postfix: str = '_text_detection_datamodel',
                  freeze_frame_sec: Optional[int] = None,
                  add_timeline=True,
                  add_tl_indicators=True,
                  # add_tl_graph=True,
                  frames_per_round: int = 100) -> None:
    """ Burns in the text detection JSON in the video and
    adds a timeline animation on the bottom displaying the detection density.

    Args:
        json_path (str): path of the JSON folder
        video_path (str): path of the folder of the video.
        v_name (str): name of the original video.
        out_path (str): folder for the output video.
        json_postfix (str, optional): postfix of the JSON file.
                                      Defaults to '_text_detection_datamodel'.
        add_timeline (bool, optional): Flag for the addition of the timeline.
                                       Defaults to True.
        add_tl_indicators (bool, optional): Flag for the timeline indicators.
                                            Defaults to True.
        freeze_frame_sec (int or None): The amount of seconds the burn-in text
                                        frame is displayed. Defaults to None,
                                        indicating the duration is one frame.
        frames_per_round (int, optional): Sets the amount of detections per
        round, can be optimized for performance. Defaults to 100.
    """
    strat = TextDetectionStrat(add_timeline,
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
