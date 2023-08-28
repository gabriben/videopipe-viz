import subprocess
import numpy as np
import moviepy.editor as mp
import matplotlib.pyplot as plt
import seaborn as sns
from moviepy.video.io.bindings import mplfig_to_npimage
import videopipeViz.core_viz as core


class TimelineAnimation:
    """ Class for the timeline animation. """
    def __init__(self,
                 clip,
                 v_name,
                 data,
                 task,
                 detection_delay_sec=0,
                 add_graph=True,  # Currently unused and always True
                 add_indicators=True,
                 ):
        self.clip = clip
        self.v_name = v_name
        self.data = data
        self.task = task
        self.detection_delay_sec = detection_delay_sec
        self.add_graph = add_graph
        self.add_indicators = add_indicators
        self.fps = clip.fps
        self.total_frames_delay = 0
        self.delay_frames_set = set(data)
        self.delay_frames_left = int(detection_delay_sec * self.fps)
        self.last_line = None

        self.file_name = None

        self.fig, self.ax = self._make_timeline()
        self.total_video_time = clip.duration + len(data) * detection_delay_sec

    def _calculate_time_indicator_frame_number(self, t: int) -> int:
        """ Helper function to calculate the frame number of the video when
        frames are delayed (frozen). This is necessary for the insertion of
        the freeze frames of shotboundaries.

        Args:
            t (int): the current timestamp.

        Returns:
            int: current frame number.
        """
        frame_number = int(round(t * self.fps))

        if self.delay_frames_left > 0:
            self.delay_frames_left -= 1
            self.total_frames_delay += 1

            if frame_number - self.total_frames_delay in self.delay_frames_set:
                self.delay_frames_set.discard(frame_number -
                                              self.total_frames_delay)
                return frame_number - self.total_frames_delay + 1

        return frame_number

    def _make_timeline(self,
                       height_ratio: int = 7,
                       detail_modifier: float = 1.0):
        """ Makes the timeline plot.

        Args:
            height_ratio (int, optional): The ratio of the video to the
            timeline. Defaults to 7 for a timeline that's 1/7 the video heigth.
            detail_modifier (float, optional): Modifier for the amount of
            detail of the timeline plot. Higher number equals more detail.
            Defaults to 1.0.

        Returns:
            fig, ax: The figure and axis of the plot.
        """
        DETAIL_SWEETSPOT = 0.03

        w, h = self.clip.size
        total_frames = int(self.clip.duration * self.fps)
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches

        fig, ax = plt.subplots(figsize=(w * px, h / height_ratio * px))

        sns.set_style('whitegrid')
        g = sns.kdeplot(np.array(self.data),
                        clip=(0, total_frames),
                        bw_method=DETAIL_SWEETSPOT * detail_modifier,
                        color='navy',
                        zorder=100)

        if self.add_indicators:
            ymin, ymax = g.get_ylim()
            g.vlines(self.data,
                     ymin=ymin,
                     ymax=ymax,
                     colors='lightblue',
                     lw=1,
                     zorder=0)

        # midroll_indicator = 10664
        # ax.axvline(midroll_indicator,
        #            color='orange',
        #            linestyle='solid',
        #            linewidth=2)

        ax.set_xlim(0, total_frames)
        axis_frames = range(0, total_frames, total_frames // 10)
        axis_timestamps = [core.frame_number_to_timestamp(fr,
                                                          self.fps,
                                                          format='seconds')
                           for fr in axis_frames]
        ax.set_xticks(axis_frames, axis_timestamps)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        return fig, ax

    def _make_frame(self, t: int):
        """ Helper function to make the timeline animation. Determines the
        the frame of the animation for all timestamps t.

        Args:
            t (int): current timestamp

        Returns:
            frame image
        """
        if self.last_line is not None:
            self.last_line.remove()

        time_indicator_frame = self._calculate_time_indicator_frame_number(t)
        self.last_line = self.ax.axvline(time_indicator_frame,
                                         color=(1, 0, 0),
                                         linestyle='dashed',
                                         linewidth=1)

        return mplfig_to_npimage(self.fig)

    def create(self, timeline_postfix='_timeline_only'):
        """ Create the timeline animation and write to a file formatted as:
        <original video name> + <task> + <timeline_postfix> + '.mp4'

        example: 'video_name_face_detection_timeline_only.mp4'

        Also saves this filename for when the add_to_video method will be
        called afterwards.

        Args:
            timeline_postfix (str, optional): postfix of the filename to
            write to. Defaults to '_timeline_only'.

        Returns:
            self
        """
        animation = mp.VideoClip(self._make_frame,
                                 duration=self.total_video_time)
        core.write_clip(animation,
                        self.v_name + self.task + timeline_postfix,
                        audio=False)
        self.file_name = self.v_name + self.task + timeline_postfix + '.mp4'

        return self

    def add_to_video(self, burned_in_video_path: str, output_filename: str):
        """ Adds the timeline animation to the video with
        the burned in detections.

        Args:
            burned_in_video_path (str): path of the video with the detections
            burned into it.
            output_filename (str): filename of the output video.

        Raises:
            ValueError: If self.file_name is not set by the create method, an
            error is raised.
        """
        if not self.file_name:
            raise ValueError('No filename known, ' +
                             'did you create the animation first?')
        cmd = f"ffmpeg -i {burned_in_video_path} -i {self.file_name} " \
              f"-filter_complex vstack {output_filename}"
        subprocess.call(cmd, shell=True)










