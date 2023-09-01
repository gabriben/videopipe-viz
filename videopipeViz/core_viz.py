from abc import abstractmethod
from typing import List, Optional, Tuple
from PIL import Image
import moviepy.editor as mp
import numpy as np
import subprocess
import os

import videopipeViz.timeline as tl


class BurnInStrategy():
    def __init__(self,
                 add_timeline: bool = True,
                 add_tl_indicators: bool = True,
                 # add_tl_graph: bool = True,
                 freeze_frame_sec: Optional[int] = None) -> None:
        self.add_timeline = add_timeline
        self.add_tl_indicators = add_tl_indicators
        # self.add_tl_graph = add_tl_graph
        self.freeze_frame_sec = (freeze_frame_sec if freeze_frame_sec != 0
                                 else None)

    @abstractmethod
    def read_json(self,
                  json_path: str,
                  v_name: str,
                  json_postfix: str) -> List[dict]:
        pass

    @abstractmethod
    def get_clips(self,
                  clip: mp.VideoFileClip,
                  detections: list,
                  inserted_frame_duration: float,
                  timestamp_offset) -> Tuple[List[mp.VideoClip], int]:
        pass


def frame_number_to_timestamp(frame_number: int,
                              fps: int,
                              format='milliseconds'):
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds % 1) * 1000)
    if format == 'milliseconds':
        timestamp = f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
    elif format == 'seconds':
        timestamp = f"{minutes:02d}:{seconds:02d}"

    return timestamp


def get_frame_by_number(clip: mp.VideoFileClip, frame_number: int) -> Image:
    """ Returns the frame from the clip by their frame_number. """
    frame_duration = 1 / clip.fps
    frame = clip.get_frame(frame_number * frame_duration)
    return Image.fromarray(frame)


def read_clip(v_name: str) -> mp.VideoFileClip:
    ''' Read the video file at given path. '''
    return mp.VideoFileClip(v_name + '.mp4')


def scale_bb_to_image(clip: mp.VideoFileClip,
                      y0: int,
                      x1: int,
                      y1: int,
                      x0: int,
                      RESIZE_DIM: int = 640) -> Tuple[int, int, int, int]:
    """ Scales a bounding box to the image
        using the RESIZE_DIM variable.
    """

    w, h = clip.size
    width_ratio = w / RESIZE_DIM
    height_ratio = h / RESIZE_DIM

    y0 = int(y0 * height_ratio)
    y1 = int(y1 * height_ratio)
    x0 = int(x0 * width_ratio)
    x1 = int(x1 * width_ratio)

    return (x0, y0, x1, y1)


def create_text_clip(txt: str,
                     txtclip_dur: float = 1/25,
                     color: str = 'white',
                     font: str = "Century-Schoolbook-Roman",
                     fontsize: int = 70,
                     kerning: int = -2,
                     interline: int = -1,
                     bg_color: str = 'black',
                     clipsize: Tuple[int, int] = (1920, 1080)) -> mp.TextClip:
    '''
    Create frame of length txtclip_dur with txt in the center.
    '''
    txtclip = (mp.TextClip(txt, color=color,
               font=font, fontsize=fontsize, kerning=kerning,
               interline=interline, bg_color=bg_color, size=clipsize)
               .set_duration(txtclip_dur)
               .set_position(('center')))
    return txtclip


def create_top_frame_clip(clip: mp.VideoFileClip,
                          top_frames: list,
                          still_duration: int = 3,
                          text_frame_duration: int = 1):
    '''
    Returns a clip of the top frames preceded by a clip showing the ranking of
    the clip.
    '''

    start_frame = create_text_clip(f"Top {len(top_frames)} thumbnails",
                                   text_frame_duration)
    clips = [start_frame]
    count = len(top_frames)
    for frame in top_frames:
        txtclip = create_text_clip(f"{count}.", text_frame_duration)
        still = get_frame_by_number(clip, frame)
        imgclip = mp.ImageClip(np.asarray(still), duration=still_duration)
        clips.append(txtclip)
        clips.append(imgclip)
        count -= 1

    return clips


def write_audioclip(clip: mp.VideoFileClip, v_name: str, logger=None) -> None:
    ''' Split audio from clip and write it to a file. '''
    audio = clip.audio
    audio.write_audiofile(v_name + '_audio.mp3', logger=logger)


def write_clip(clip: mp.VideoFileClip,
               name: str,
               postfix: str = '',
               audio=True,
               fps: int = 25,
               logger=None) -> None:
    ''' Write the clip to a file. Try using hw acceleration first. '''
    if postfix != '':
        postfix = '_' + postfix
    try:
        clip.write_videofile(f"{name}{postfix}.mp4",
                             codec='h264_nvenc',
                             fps=fps,
                             logger=logger,
                             audio=audio,
                             preset='fast')
    except IOError:
        try:
            clip.write_videofile(f"{name}{postfix}.mp4",
                                 codec='libx264',
                                 fps=fps,
                                 logger=logger,
                                 audio=audio,
                                 preset='ultrafast')
        except IOError:
            raise Exception('An error occured while writing the video file.')


def files_to_video(clip: mp.VideoFileClip,
                   v_name: str,
                   rounds: int,
                   filename: str,
                   output_name: str,
                   overwrite_audio=True) -> None:
    ''' Concatenate the video files in filename and write it to output_name.
        If retain_audio is True, the audio will be added to the video. '''
    if overwrite_audio:
        # First write to temp.mp4 then add audio and write to output_name.
        concatenate_videofiles(filename, "temp.mp4")
        write_audioclip(clip, v_name)
        add_audio_to_video("temp.mp4", v_name + '_audio.mp3', output_name)
    else:
        concatenate_videofiles(filename, output_name)

    clean_up_files(v_name, rounds, filename)


def concatenate_videofiles(filename: str, output_name: str) -> None:
    ''' Concatenate the video files in filename
        and write it to output_name.
    '''
    cmd = f"ffmpeg -f concat -safe 0 -i {filename} -c copy -y {output_name}"
    subprocess.call(cmd, shell=True)


def add_audio_to_video(video_name: str, audio_name: str, output_name: str):
    ''' Add audio to video and write it to output_name. '''
    cmd = (f"ffmpeg -i {video_name} -i {audio_name} "
           f"-map 0 -map 1:a -c:v copy -shortest {output_name}")
    subprocess.call(cmd, shell=True)


def clean_up_files(v_name: str,
                   rounds: int,
                   txt_filename: str,
                   output_filename: str = 'temp.mp4'):
    ''' Clean up the files created during the process. '''
    # Delete all the subclips.
    for i in range(rounds + 1):
        os.remove(v_name + '_' + str(i) + '.mp4')

    # Delete the face_detection.txt file.
    os.remove(txt_filename)

    # Delete the audio file.
    if os.path.exists(v_name + '_audio.mp3'):
        os.remove(v_name + '_audio.mp3')

    if os.path.exists(output_filename):
        os.remove(output_filename)


def burn_in_video(strat: BurnInStrategy,
                  json_path: str,
                  video_path: str,
                  v_name: str,
                  out_path: str,
                  json_postfix: str,
                  frames_per_round: int = 100) -> None:
    """ Burns in the detection JSON in the video and
    adds a timeline animation on the bottom displaying the detection density.

    Args:
        json_path (str): path of the JSON folder
        video_path (str): path of the folder of the video.
        v_name (str): name of the original video.
        out_path (str): folder for the output video.
        json_postfix (str, optional): postfix of the JSON file.
        frames_per_round (int, optional): Sets the amount of detections per
        round, can be optimized for performance. Defaults to 100.
    """
    detections = strat.read_json(json_path, v_name, json_postfix)
    clip = read_clip(video_path + v_name)

    # If we insert more than one frame, we cannot overwrite the burned in video
    # with the original audio anymore.
    retain_audio = strat.freeze_frame_sec is not None

    # If freeze_frame_sec is not set, use the duration of one frame.
    inserted_frame_duration = (strat.freeze_frame_sec if strat.freeze_frame_sec
                               else 1 / clip.fps)
    total_detects = len(detections)
    total_rounds = total_detects // frames_per_round

    prev_ts = 0
    with open(out_path + 'temp_video_files_log.txt', 'w') as f:
        print(f"Started editing {total_detects} frames.")
        for round in range(total_rounds + 1):
            clips = []
            start_frame_number = round * frames_per_round
            end_frame_number = start_frame_number + frames_per_round
            frame_batch = detections[start_frame_number:end_frame_number]
            clips, prev_ts = strat.get_clips(clip,
                                             frame_batch,
                                             inserted_frame_duration,
                                             prev_ts)

            if (round == total_rounds):
                clips.append(clip.subclip(prev_ts, clip.duration))

            write_clip(mp.concatenate_videoclips(clips),
                       v_name,
                       postfix=str(round),
                       audio=retain_audio)
            f.write('file ' + v_name + '_' + str(round) + '.mp4\n')
            print(f"{min(total_detects, (round + 1) * frames_per_round)}/" +
                  f"{total_detects}")

    new_v_name = out_path + v_name + json_postfix

    print("Started concatenating temporary video files.")
    files_to_video(clip,
                   v_name,
                   total_rounds,
                   out_path + 'temp_video_files_log.txt',
                   new_v_name + '.mp4',
                   overwrite_audio=(not retain_audio))
    print(f"Concatenation completed and written to {new_v_name}.mp4")

    if not strat.add_timeline:
        return

    frame_numbers = [d['dimension_idx'] for d in detections]
    print("Started making the timeline animation.")
    timeline = tl.TimelineAnimation(clip,
                                    v_name,
                                    frame_numbers,
                                    # add_graph=strat.add_tl_graph,
                                    add_indicators=strat.add_tl_indicators,
                                    delay_sec=inserted_frame_duration,
                                    task=json_postfix)
    print("Adding the timeline animation to the burned-in video.")
    timeline.add_to_video(new_v_name + '.mp4', new_v_name + '_timeline.mp4')

    if os.path.exists(new_v_name + '.mp4'):
        os.remove(new_v_name + '.mp4')

    print(f"Finished! Final video written to {new_v_name}_timeline.mp4")
