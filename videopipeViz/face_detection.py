from typing import Optional, Tuple
import pandas as pd
import numpy as np
import moviepy.editor as mp
from PIL import ImageDraw, Image
import videopipeViz.core_viz as core


class FaceDetectionStrat(core.BurnInStrategy):
    def __init__(self,
                 add_timeline: bool,
                 add_tl_indicators: bool,
                 # add_tl_graph: bool,
                 freeze_frame_sec: Optional[int] = None):
        super().__init__(add_timeline,
                         add_tl_indicators,
                         # add_tl_graph,
                         freeze_frame_sec)

    def read_json(self, json_path: str, v_name: str, json_postfix: str):
        '''
        Read the face detection JSON file.
        '''
        faces = pd.read_json(json_path + v_name + '/'
                             + v_name + json_postfix + '.json',
                             lines=True)
        faces_detected = [f for f in faces.data[0] if len(f['faces']) > 0]
        return faces_detected

    def get_clips(self,
                  clip: mp.VideoFileClip,
                  faces_detected: list,
                  face_frame_duration: float,
                  timestamp_offset: int = 0):
        """
        Make a list of clips with all the face frames in 'faces_detected'
        inserted in 'clip'. face_frames are inserted with a duration of
        'face_frame_duration'. 'timestamp_offset' is used to determine the
        starting time of the first (faceless) subclip.
        """

        clips = []
        for faces in faces_detected:
            ts, bb_frame = self._make_frame(clip, faces)

            if (timestamp_offset != ts):
                clips.append(clip.subclip(timestamp_offset, ts))

            face_frame_clip = mp.ImageClip(np.asarray(bb_frame),
                                           duration=face_frame_duration)
            clips.append(face_frame_clip)
            timestamp_offset = ts + face_frame_duration

        return clips, timestamp_offset

    def _draw_bounding_boxes(self,
                             clip: mp.VideoFileClip,
                             faces: dict,
                             img: Image.Image,
                             color: str = 'red',
                             bb_width: int = 5) -> Image.Image:
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

    def _make_frame(self,
                    clip: mp.VideoFileClip,
                    faces: dict) -> Tuple[float, Image.Image]:
        """ Draw the faces on top of the frame in 'clip' and
            also return the corresponding frame timestamp. """
        face_frame_number = faces['dimension_idx']
        face_timestamp = face_frame_number / clip.fps
        frame = core.get_frame_by_number(clip, face_frame_number)
        bb_frame = self._draw_bounding_boxes(clip, faces, frame)

        return face_timestamp, bb_frame


def faceDetection(json_path: str,
                  video_path: str,
                  v_name: str,
                  out_path: str,
                  json_postfix: str = '_face_detection_datamodel',
                  freeze_frame_sec: Optional[int] = None,
                  add_timeline=True,
                  add_tl_indicators=True,
                  # add_tl_graph=True,
                  frames_per_round: int = 100) -> None:
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
        freeze_frame_sec (int or None): The amount of seconds the burn-in face
                                        frame is displayed. Defaults to None,
                                        indicating the duration is one frame.
        faces_per_round (int, optional): Sets the amount of detections per
        round, can be optimized for performance. Defaults to 100.
    """
    strat = FaceDetectionStrat(add_timeline,
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
