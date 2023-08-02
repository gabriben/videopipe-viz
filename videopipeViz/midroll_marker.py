import pandas as pd
import numpy as np
import videopipeViz.core_viz as core
from PIL import Image, ImageDraw, ImageFont


def get_frame_by_number(clip, frame_number):
    """ Returns the frame from the clip by their frame_number. """

    frame_duration = 1 / clip.fps
    frame = clip.get_frame(frame_number * frame_duration)
    return Image.fromarray(frame)


def make_frame_line(clip, midroll_marker):
    """ Make a row of frames indicating
    the frame-precise position of the midroll. """
    w, h = clip.size

    # If the midroll_marker in the JSON is an integer,
    # it indicates the frame after the midroll.
    # If the marker is a float e.g. '4.5' this indicates
    # that the midroll should be between frames 4 and 5.
    try:
        frame_after_midroll = int(midroll_marker)
        frame_before_midroll = frame_after_midroll - 1
    except ValueError:
        frame_before_midroll = int(float(midroll_marker))
        frame_after_midroll = int(np.ceil(float(midroll_marker)))

    frame_line = Image.new('RGB', (5 * w, h))

    frame_line.paste(get_frame_by_number(clip, frame_before_midroll - 1), (0, 0))
    frame_line.paste(get_frame_by_number(clip, frame_before_midroll), (w, 0))

    font = ImageFont.truetype("NotoSansMono-Bold.ttf", 50)
    draw = ImageDraw.Draw(frame_line)
    draw.text((2.2 * w, h/3), f"Midroll between frames {frame_before_midroll} and {frame_after_midroll}", font=font, fill='white')
    draw.text((2.2 * w, h/2), f"timestamps: {frame_before_midroll/clip.fps} and {frame_after_midroll/clip.fps}", font=font, fill='white')

    frame_line.paste(get_frame_by_number(clip, frame_after_midroll), (3 * w, 0))
    frame_line.paste(get_frame_by_number(clip, frame_after_midroll + 1), (4 * w, 0))

    return frame_line


def midrollMarker(json_path: str,
                  video_path: str,
                  v_name: str,
                  out_path: str) -> None:
    # read thumbnail json
    
    midroll = pd.read_json(json_path + v_name + '/' + v_name + '_midroll_marker_output' + '.json', lines=True)
    midroll_markers = midroll['midroll_markers'][0]

    # Read video file with moviepy
    clip = core.read_clip(video_path + v_name)
    # mp.VideoFileClip(video_path + v_name + '.mp4')

    make_frame_line(clip, midroll_markers[0]).save(out_path + v_name + "_midroll_indication.jpg")
    

# if __name__ == '__main__':

#     # this can be empty if the video file and its videopipe output are at the same
#     # location as the code
#     path = ''
#     video_path = 'Videos/'
#     v_name = 'HIGH_LIGHTS_I_SNOWMAGAZINE_I_SANDER_26'
#     task = '_midroll_marker_output'

#     # read thumbnail json

#     midroll = pd.read_json(f"{path + v_name}/{v_name + task}.json", lines=True)
#     midroll_markers = midroll['midroll_markers'][0]

#     # Read video file with moviepy

#     clip = core.read_clip(v_name)
#     # mp.VideoFileClip(video_path + v_name + '.mp4')

#     make_frame_line(clip, midroll_markers[0]).save(f"{v_name}_midroll_indication.jpg")
