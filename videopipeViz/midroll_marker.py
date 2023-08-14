import pandas as pd
import numpy as np
import videopipeViz.core_viz as core
from PIL import Image, ImageDraw, ImageFont
from urllib.request import urlopen


def make_frame_line(clip, midroll_marker, surrounding_frames=2,
                    frame_type='Midroll'):
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

    total_frames = 2 * surrounding_frames + 1
    frame_line = Image.new('RGB', (total_frames * w, h))

    for before_idx in range(surrounding_frames):
        frame = core.get_frame_by_number(clip,
                                         frame_before_midroll - before_idx)
        pos_in_line = ((surrounding_frames - 1 - before_idx) * w, 0)
        frame_line.paste(frame, pos_in_line)

    for after_idx in range(surrounding_frames):
        frame = core.get_frame_by_number(clip, frame_after_midroll + after_idx)
        pos_in_line = ((2 * surrounding_frames - after_idx) * w, 0)
        frame_line.paste(frame, pos_in_line)

    #font = ImageFont.truetype("NotoSansMono-Bold.ttf", 70)
    truetype_url = 'https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Medium.ttf?raw=true'
    font = ImageFont.truetype(urlopen(truetype_url), size=10)
    draw = ImageDraw.Draw(frame_line)
    output_text = f"{frame_type.capitalize()} between frames " + \
                  f"{frame_before_midroll} and {frame_after_midroll}"
    draw.text(((surrounding_frames + 0.05) * w, h/3),
              output_text,
              font=font,
              fill='white')
    before_timestamp = core.frame_number_to_timestamp(frame_before_midroll,
                                                      clip.fps)
    after_timestamp = core.frame_number_to_timestamp(frame_after_midroll,
                                                     clip.fps)
    draw.text(((surrounding_frames + 0.05) * w, h/2),
              f"timestamps: {before_timestamp} and {after_timestamp}",
              font=font,
              fill='white')

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
