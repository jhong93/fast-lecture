# Lecture 2.0 X

Watch videos at higher speed by skipping silence.

This speeds up lecture videos by about 1.5x. Since the speedup is done by skipping pauses and silence, the words are not distorted. Speed up the video by another 1.5 - 2x in your favorite player (YMMV).

Internally, the script uses OpenAI Whisper to generate word aligned captions. A GPU will make the process faster.

## Usage

Install `ffmpeg`, which is needed for `moviepy`.

Install Python dependencies: `pip install -r requirements.txt`

Run the script: `python retime.py [input.mp4] -o [output.mp4]`

This will write a new video file with silence removed.

## License

See [LICENSE.md](LICENSE.md).