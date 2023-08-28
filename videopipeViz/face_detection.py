import os
import pandas as pd
import numpy as np
import moviepy.editor as mp
from PIL import ImageDraw

import videopipeViz.core_viz as core
import videopipeViz.timeline as tl


def read_face_detection(json_path, v_name, json_postfix):
    '''
    Read the face detection JSON file.
    '''
    faces = pd.read_json(json_path + v_name + '/'
                         + v_name + json_postfix + '.json',
                         lines=True)
    faces_detected = [f for f in faces.data[0] if len(f['faces']) > 0]
    return faces_detected


def draw_bounding_boxes(clip, faces, img, color='red', bb_width=5):
    '''
    Draw all bounding boxes on the detected faces of the image.

    clip: movie clip
    faces: list of detected faces
    img: frame on which we draw the bounding boxes
    color: color of the bounding box
    bb_width: width of the bounding box

    '''
    for face in faces['faces']:
        scaled_bb = core.scale_bb_to_image(clip, *face['bb_faces'])
        draw = ImageDraw.Draw(img)
        draw.rectangle(scaled_bb, outline=color, width=bb_width)

    return img


def make_frame(clip, faces):
    """ Draw the faces on top of the frame in 'clip' and
        also return the corresponding frame timestamp. """
    face_frame_number = faces['dimension_idx']
    face_timestamp = face_frame_number / clip.fps
    frame = core.get_frame_by_number(clip, face_frame_number)
    bb_frame = draw_bounding_boxes(clip, faces, frame)

    return face_timestamp, bb_frame


def get_face_clips(clip, faces_detected, face_frame_duration,
                   timestamp_offset=0):
    """
    Make a list of clips with all the face frames in 'faces_detected'
    inserted in 'clip'. face_frames are inserted with a duration of
    'face_frame_duration'. 'timestamp_offset' is used to determine the
    starting time of the first (faceless) subclip.
    """

    clips = []
    for faces in faces_detected:
        ts, bb_frame = make_frame(clip, faces)

        if (timestamp_offset != ts):
            clips.append(clip.subclip(timestamp_offset, ts))

        face_frame_clip = mp.ImageClip(np.asarray(bb_frame),
                                       duration=face_frame_duration)
        clips.append(face_frame_clip)
        timestamp_offset = ts + face_frame_duration

    return clips, timestamp_offset


def faceDetection(json_path: str,
                  video_path: str,
                  v_name: str,
                  out_path: str,
                  json_postfix: str = '_face_detection_datamodel',
                  add_timeline=True,
                  add_tl_indicators=True,
                  add_tl_graph=True,
                  faces_per_round: int = 100) -> None:
    """ Burns in the face detection JSON in the video and
    adds a timeline animation on the bottom displaying the detection density.

    Args:
        json_path (str): path of the JSON folder
        video_path (str): path of the folder of the video.
        v_name (str): name of the original video.
        out_path (str): folder for the output video.
        json_postfix (str, optional): postfix of the JSON file.
                                      Defaults to '_face_detection_datamodel'.
        add_timeline (bool, optional): Flag for the addition of the timeline.
                                       Defaults to True.
        add_tl_indicators (bool, optional): Flag for the timeline indicators.
                                            Defaults to True.
        add_tl_graph (bool, optional): Flag for the addition of the graphline.
                                       Currently UNUSED. Defaults to True.
        faces_per_round (int, optional): Sets the amount of detections per
        round, can be optimized for performance. Defaults to 100.
    """
    faces_detected = read_face_detection(json_path, v_name, json_postfix)
    clip = core.read_clip(video_path + v_name)
    frame_duration = 1 / clip.fps

    face_frame_duration = frame_duration
    total_rounds = len(faces_detected) // faces_per_round

    prev_ts = 0
    with open(out_path + 'face_detection.txt', 'w') as f:
        # Create video clips with 'faces_per_round' amount of detected faces
        # inserted per clip.
        for round in range(total_rounds + 1):
            clips = []
            start_face_number = round * faces_per_round
            end_face_number = start_face_number + faces_per_round
            face_batch = faces_detected[start_face_number:end_face_number]
            clips, prev_ts = get_face_clips(clip,
                                            face_batch,
                                            face_frame_duration,
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
                        out_path + 'face_detection.txt',
                        new_v_name + '.mp4')

    if not add_timeline:
        return

    face_frame_numbers = [face['dimension_idx'] for face in faces_detected]
    timeline = tl.TimelineAnimation(clip,
                                    v_name,
                                    face_frame_numbers,
                                    add_graph=add_tl_graph,
                                    add_indicators=add_tl_indicators,
                                    detection_delay_sec=0,
                                    task=json_postfix)

    timeline.add_to_video(new_v_name + '.mp4', new_v_name + '_timeline.mp4')

    if os.path.exists(new_v_name + '.mp4'):
        os.remove(new_v_name + '.mp4')
