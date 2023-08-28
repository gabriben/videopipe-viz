import os
import pandas as pd
import moviepy.editor as mp
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import videopipeViz.core_viz as core
import videopipeViz.timeline as tl


def read_text_detection(json_path, v_name, json_postfix):
    '''
    Read the text detection JSON file.
    '''
    text = pd.read_json(json_path + v_name + '/'
                        + v_name + json_postfix + '.json',
                        lines=True)
    texts_detected = [f for f in text.data[0] if len(f['text']) > 0]
    return texts_detected


def draw_text_bb(frame, texts, color='blue', bb_width=5, txtcolor='black'):
    """ Draw all the detected text in 'texts' on top of the frame. """
    copy = frame.copy()
    font = ImageFont.truetype("NotoSansMono-Bold.ttf", 20)
    for text in texts:
        left, top, width, height, conf, detected_text = texts[text].values()
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


def make_frame(clip, txts):
    """
    Get the frame in 'txts' and draw all the texts in 'txts' on the frame.
    Also return the timestamp in the clip of the detected frame.
    """
    txt_frame_number = txts['dimension_idx']
    txt_timestamp = txt_frame_number / clip.fps
    frame = core.get_frame_by_number(clip, txt_frame_number)
    bb_frame = draw_text_bb(frame, txts['text'])

    return txt_timestamp, bb_frame


def get_txt_clips(clip,
                  texts_detected,
                  txt_frame_duration,
                  timestamp_offset=0):
    """ Make a list of clips with all the text frames in 'texts_detected'.
    Text frames are inserted with a duration of 'txt_frame_duration'.
    'timestamp_offset' is used to determine the starting time of the first
    (textless) subclip.
    """
    clips = []
    for txt in texts_detected:
        ts, bb_frame = make_frame(clip, txt)

        if (timestamp_offset != ts):
            clips.append(clip.subclip(timestamp_offset, ts))

        txt_frame_clip = mp.ImageClip(np.asarray(bb_frame),
                                      duration=txt_frame_duration)
        clips.append(txt_frame_clip)
        timestamp_offset = ts + txt_frame_duration

    return clips, timestamp_offset


def textDetection(json_path: str,
                  video_path: str,
                  v_name: str,
                  out_path: str,
                  json_postfix: str = '_text_detection_datamodel',
                  add_timeline=True,
                  add_tl_indicators=True,
                  add_tl_graph=True,
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
        add_tl_graph (bool, optional): Flag for the addition of the graphline.
                                       Currently UNUSED. Defaults to True.
        frames_per_round (int, optional): Sets the amount of detections per
        round, can be optimized for performance. Defaults to 100.
    """
    text_detected = read_text_detection(json_path, v_name, json_postfix)
    clip = core.read_clip(video_path + v_name)
    frame_duration = 1 / clip.fps

    txt_frame_duration = frame_duration
    total_rounds = len(text_detected) // frames_per_round

    prev_ts = 0
    with open(out_path + 'text_detection.txt', 'w') as f:
        for round in range(total_rounds + 1):
            clips = []
            start_txt_number = round * frames_per_round
            end_txt_number = start_txt_number + frames_per_round
            txt_batch = text_detected[start_txt_number:end_txt_number]
            clips, prev_ts = get_txt_clips(clip,
                                           txt_batch,
                                           txt_frame_duration,
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
                        out_path + 'text_detection.txt',
                        new_v_name + '.mp4')

    if not add_timeline:
        return

    text_frame_numbers = [txt['dimension_idx'] for txt in text_detected]
    timeline = tl.TimelineAnimation(clip,
                                    v_name,
                                    text_frame_numbers,
                                    add_graph=add_tl_graph,
                                    add_indicators=add_tl_indicators,
                                    detection_delay_sec=0,
                                    task=json_postfix)

    timeline.add_to_video(new_v_name + '.mp4', new_v_name + '_timeline.mp4')

    if os.path.exists(new_v_name + '.mp4'):
        os.remove(new_v_name + '.mp4')
