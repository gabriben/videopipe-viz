from PIL import Image
import moviepy.editor as mp
from numpy import asarray
import subprocess
import os

# Change the ffmpeg binary of moviepy to the local one to allow for hw acceleration.
try:
    from moviepy.config import change_settings
    change_settings({"FFMPEG_BINARY":"ffmpeg"})
except:
    pass


def get_frame_by_number(clip, frame_number):
    """ Returns the frame from the clip by their frame_number. """
    frame_duration = 1 / clip.fps
    frame = clip.get_frame(frame_number * frame_duration)
    return Image.fromarray(frame)


def read_clip(v_name):
    ''' Read the video file at given path. '''
    clip = mp.VideoFileClip(v_name + '.mp4')
    return clip


def scale_bb_to_image(clip, y0, x1, y1, x0, RESIZE_DIM=640):
    """ Scales a bounding box to the image using the global RESIZE_DIM variable. """

    w, h = clip.size
    width_ratio = w / RESIZE_DIM
    height_ratio = h / RESIZE_DIM

    y0 = int(y0 * height_ratio)
    y1 = int(y1 * height_ratio)
    x0 = int(x0 * width_ratio)
    x1 = int(x1 * width_ratio)

    return [x0, y0, x1, y1]


def create_text_clip(txt, txtclip_dur=1/25, color='white', font="Century-Schoolbook-Roman", fontsize=70, kerning=-2, interline=-1, bg_color='black', clipsize = (1920, 1080)):
    '''
    Create frame of length txtclip_dur with txt in the center.
    '''
    txtclip = (mp.TextClip(txt, color=color,
               font=font, fontsize=fontsize, kerning=kerning,
               interline=interline, bg_color=bg_color, size=clipsize)
               .set_duration(txtclip_dur)
               .set_position(('center')))
    return txtclip


def create_top_frame_clip(clip, top_frames, still_duration=3, text_frame_duration=1):
    '''
    Returns a clip of the top frames preceded by a clip showing the ranking of
    the clip.
    '''

    start_frame = create_text_clip(f"Top {len(top_frames)} thumbnails", text_frame_duration)
    clips = [start_frame]
    count = len(top_frames)
    for frame in top_frames:
        txtclip = create_text_clip(f"{count}.", text_frame_duration)
        still = get_frame_by_number(clip, frame)
        imgclip = mp.ImageClip(asarray(still), duration=still_duration)
        clips.append(txtclip)
        clips.append(imgclip)
        count -= 1

    return clips


def write_audioclip(clip, v_name, logger=None):
    ''' Split audio from clip and write it to a file. '''
    audio = clip.audio
    audio.write_audiofile(v_name + '_audio.mp3', logger=logger)


def write_clip(clip, name, postfix='', audio=True, fps=25, logger=None):
    ''' Write the clip to a file. Try using hw acceleration first. '''
    try:
        clip.write_videofile(name + '_' + postfix + '.mp4', codec='h264_nvenc', fps=fps, logger=logger, audio=audio, preset='fast')
    except:
        try:
            clip.write_videofile(name + '_' + postfix + '.mp4', codec='libx264', fps=fps, logger=logger, audio=audio, preset='ultrafast')
        except:
            raise Exception('An error occured while writing the video file.')

def files_to_video(clip, v_name, rounds, filename, output_name, retain_audio=True):
    ''' Concatenate the video files in filename and write it to output_name.
        If retain_audio is True, the audio will be added to the video. '''
    if retain_audio:
        # First write to temp.mp4 then add audio and write to output_name.
        concatenate_videofiles(filename, "temp.mp4")
        write_audioclip(clip, v_name)
        add_audio_to_video("temp.mp4", v_name + '_audio.mp3', output_name)
    else:
        concatenate_videofiles(filename, output_name)

    clean_up_files(v_name, rounds, filename)


def concatenate_videofiles(filename, output_name):
    ''' Concatenate the video files in filename and write it to output_name. '''
    subprocess.call("ffmpeg -f concat -safe 0 -i " + filename + " -c copy -y " + output_name, shell=True)


def add_audio_to_video(video_name, audio_name, output_name):
    ''' Add audio to video and write it to output_name. '''
    subprocess.call("ffmpeg -i " + video_name + " -i " + audio_name + " -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -y " + output_name, shell=True)


def clean_up_files(v_name, rounds, txt_filename, output_filename='temp.mp4'):
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