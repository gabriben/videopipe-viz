import pandas as pd
import moviepy.editor as mp
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import core_viz as core


def read_text_detection(path, v_name, task):
    '''
    Read the text detection JSON file.
    '''
    text = pd.read_json(path + v_name + '/' + v_name + task + '.json', lines = True)
    texts_detected = [f for f in text.data[0] if len(f['text']) > 0]
    return texts_detected


def draw_text_bb(frame, texts, color='blue', bb_width=5, txtcolor='black'):
    """ Draw all the detected text in 'texts' on top of the frame. """
    copy = frame.copy()
    for text in texts:
        left, top, width, height, conf, detected_text = texts[text].values()
        right = left + width
        bottom = top + height
        draw = ImageDraw.Draw(copy)
        draw.rectangle([left, bottom, right, top], outline=color, width=bb_width)
        draw.text((left, bottom), detected_text + "(" + str(conf) + ")", font=font, fill=txtcolor)
    return copy


def make_frame(clip, txts):
    """ Get the frame in 'txts' and draw all the texts in 'txts' on top of the frame.
    Also return the timestamp in the clip of the detected frame.
    """
    txt_frame_number = txts['dimension_idx']
    txt_timestamp = txt_frame_number / clip.fps
    frame = core.get_frame_by_number(clip, txt_frame_number)
    bb_frame = draw_text_bb(frame, txts['text'])

    return txt_timestamp, bb_frame


def get_txt_clips(clip, texts_detected, txt_frame_duration, timestamp_offset=0):
    """ Make a list of clips with all the text frames in 'texts_detected' inserted in 'clip'.
    text frames are inserted with a duration of 'txt_frame_duration'.
    'timestamp_offset' is used to determine the starting time of the first (textless) subclip.
    """
    clips = []
    for txt in texts_detected:
        ts, bb_frame = make_frame(clip, txt)

        if (timestamp_offset != ts):
            clips.append(clip.subclip(timestamp_offset, ts))

        txt_frame_clip = mp.ImageClip(np.asarray(bb_frame), duration=txt_frame_duration)
        clips.append(txt_frame_clip)
        timestamp_offset = ts + txt_frame_duration

    return clips, timestamp_offset


if __name__ == '__main__':
    video_path = 'Videos/'
    v_name = 'HIGH_LIGHTS_I_SNOWMAGAZINE_I_SANDER_26'
    task = '_frame_text_detection_datamodel'
    RESIZE_DIM = 640
    output_filename = v_name + task + '.mp4'
    duration_t = 1/25

    # Requires font in /usr/share/fonts/truetype.
    font = ImageFont.truetype("NotoSansMono-Bold.ttf", 20)

    txts_detected = read_text_detection(video_path, v_name, task)

    v_name = video_path + v_name

    clip = core.read_clip(v_name)
    fps = clip.fps
    frame_duration = 1 / fps

    txts_per_round = 100
    txt_frame_duration = frame_duration
    prev_ts = 0

    f = open('txt_detection.txt', 'w')
    # Create video clips with 'faces_per_round' amount of detected faces per clip.
    for round in range(len(txts_detected) // txts_per_round + 1):
        clips = []
        start_txt_number = round * txts_per_round
        end_txt_number = start_txt_number + txts_per_round
        txt_batch = txts_detected[start_txt_number:end_txt_number]
        clips, prev_ts = get_txt_clips(clip, txt_batch, txt_frame_duration, prev_ts)

        if (round == len(txts_detected) // txts_per_round):
            clips.append(clip.subclip(prev_ts, clip.duration))

        core.write_clip(mp.concatenate_videoclips(clips), v_name, str(round), False)
        f.write('file ' + v_name + '_' + str(round) + '.mp4\n')
    f.close()

    core.files_to_video(clip, v_name, round, 'txt_detection.txt', output_filename)
