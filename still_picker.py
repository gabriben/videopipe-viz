import pandas as pd
import core_viz as core
from PIL import Image, ImageFont, ImageDraw


def read_still_picker(v_name, task):
    '''
    Read the JSON file with the still picker data.
    '''
    thumbnail = pd.read_json(v_name + '/' + v_name + task + '.json',
                             lines=True)
    thumbnail_frames = [f for f in thumbnail.thumbnails_by_frameindex]
    return thumbnail_frames


def top_still_frames(thumbnail_frames, frame_amt=5):
    '''
    Returns a list of the top frame ids based on their rank in the JSON file
    in descending order.
    '''
    # Only get the value of each key
    frames = [{'rank': v['rank'], 'frame': v['frame']}
              for d in thumbnail_frames for _, v in d.items()]

    # Sort frames based on rank
    frames = sorted(frames, key=lambda i: i['rank'])
    frames = [frame_id['frame'] for frame_id in frames[:frame_amt]]
    return frames


def make_single_still_image(clip, frame_number, rank, size,
                            number_shadow=True, border_px=0):
    frame = core.get_frame_by_number(clip, frame_number)
    font = ImageFont.truetype("NotoSansMono-Bold.ttf", 70)
    draw = ImageDraw.Draw(frame)
    text = str(rank)
    text_pos = (30, 20)
    if number_shadow:
        draw_text_shadow(draw, text, text_pos, font)

    draw.text(text_pos, text, font=font, fill='white')
    w, h = size

    return frame.resize((w-border_px, h-border_px), Image.LANCZOS)


def draw_text_shadow(draw, text, pos, font, shadowcolor='black'):
    x, y = pos
    # thin border
    draw.text((x-1, y), text, font=font, fill=shadowcolor)
    draw.text((x+1, y), text, font=font, fill=shadowcolor)
    draw.text((x, y-1), text, font=font, fill=shadowcolor)
    draw.text((x, y+1), text, font=font, fill=shadowcolor)

    # thicker border
    draw.text((x-1, y-1), text, font=font, fill=shadowcolor)
    draw.text((x+1, y-1), text, font=font, fill=shadowcolor)
    draw.text((x-1, y+1), text, font=font, fill=shadowcolor)
    draw.text((x+1, y+1), text, font=font, fill=shadowcolor)


def make_top_6_template_dict(size):
    """ Make the default template to show the top 6 best thumbnail frames.
    The dict has the format: dict[thumb_rank] = (thumb_size, thumb_pos)
    The best thumbnail is shown as the biggest in the top left,
    it is displayed as:
     _____________________
    |              |  2   |
    |       1      |______|
    |              |  3   |
    |______________|______|
    |   4   |  5   |  6   |
    |_______|______|______|

    """
    w = size[0]//3
    h = size[1]//3
    template_dict = {}
    template_dict[0] = (2*w, 2*h), (0, 0)
    template_dict[1] = (w, h),     (2*w, 0)
    template_dict[2] = (w, h),     (2*w, h)
    template_dict[3] = (w, h),     (0, 2*h)
    template_dict[4] = (w, h),     (w, 2*h)
    template_dict[5] = (w, h),     (2*w, 2*h)

    return template_dict


def make_thumbnails_image(clip, thumbnail_frames, template_dict,
                          number_shadow=True, border_px=1):

    top_frames = top_still_frames(thumbnail_frames, frame_amt=6)

    stills_image = Image.new('RGB', clip.size)

    for rank, frame in enumerate(top_frames):
        still_size, still_pos = template_dict[rank]
        still = make_single_still_image(clip,
                                        frame,
                                        rank + 1,  # Start rank from 1.
                                        still_size,
                                        number_shadow=number_shadow,
                                        border_px=border_px)
        stills_image.paste(still, still_pos)

    return stills_image


if __name__ == '__main__':

    # this can be empty if the video file and its videopipe output
    # are at the same location as the code
    path = ''
    video_path = 'Videos/'
    v_name = 'HIGH_LIGHTS_I_SNOWMAGAZINE_I_SANDER_26'
    task = '_still_picker_output'

    # Set output filename.
    output_filename = v_name + "_top_6_thumbnails.jpg"

    # Read video file.
    clip = core.read_clip(video_path + v_name)

    # Read JSON
    thumbnail_frames = read_still_picker(v_name, task)

    top_6_template = make_top_6_template_dict(clip.size)

    make_thumbnails_image(clip,
                          thumbnail_frames,
                          top_6_template,
                          border_px=2).save(output_filename)


