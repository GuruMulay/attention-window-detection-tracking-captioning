Usage info:

python a1.py <input_video> <output_video> [-persp]

Description:

The program is used to apply affine tranformation on a bounding box of user's choice that moves at a constant velocity. The program also allows applying perspective transformation using the optional '-persp' switch.

The input video must be encoded using Theora video codec (in .OGV container). A helpful website for this conversion is http://www.convertio.co . You could try ffmpeg, but we faced some glitches during playback of video output by ffmpeg.
