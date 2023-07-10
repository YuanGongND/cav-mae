# Implementation by tao88

## Data Preparation
### Download a video from Youtube (on MacOS)
Install `Homebrew` https://brew.sh/, the Missing Package Manager for macOS (or Linux)
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Install `youtube-dl` through Homebrew. The `youtube-dl` repo can be found at https://github.com/ytdl-org/youtube-dl
```
brew install youtube-dl
```
Install `ffmpeg` through Homebrew
```
brew install ffmpeg
```
Navigate to the Downloads folder
```
cd ~/Downloads
```
Download the mp4 file in the best video quality. Use the social media URL of the YT video in the share page.
```
youtube-dl -f mp4/best -k -x <video-URL> (eg. https://youtu.be/uZOcBRa2YwI)
```
If there's an issue downloading YouTube video, try:
```
brew update
brew upgrade youtube-dl
```
Example video (cars don't fly)
https://drive.google.com/file/d/1g69qziYaRUKLObjnvpNvaSJV-tBVsvNe/view?usp=drive_link

### Split the video into 10-second clip pieces
```
cd src/preprocess
python video_splitter.py -f ./preprocess/sample_video/carsdontfly.mp4 -s 10
```

### Extract audio and video frames from the mp4 video
```
cd src/preprocess
python extract_video_frame.py -input_file_list sample_video_extract_list.csv -target_fold ./sample_frames
python extract_audio.py -input_file_list sample_video_extract_list.csv -target_fold ./sample_audio
```

## Evaluating with a Pretrained Model
### Build a virtual environment and install packages
```
git clone https://github.com/ChunTao1999/cav-mae.git
python3 -m venv venv
source venv/bin/activate
TMPDIR=/var/tmp pip3 install -r requirements_a5.txt --extra-index-url https://download.pytorch.org/whl/cu116 
```

### Build sox package from source
https://ziangzhou.com/blog/Install-Sox-Locally/


## Contact
If you have a question, please bring up an issue (preferred) or send me an email [tao88@purdue.edu].
