# Implementation by tao88

## Data Preparation
### Download a video from Youtube (on MacOS)
Install Homebrew: The Missing Package Manager for macOS (or Linux)
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Install `youtube-dl` through Homebrew. The `youtube-dl` repo can be found at https://github.com/ytdl-org/youtube-dl
```
brew install youtube-dl
```
Install `ffmpeg` through Homebrew.
```
brew install ffmpeg
```
Navigate to the Downloads folder
```
cd ~/Downloads
```
Download the mp4 file in the best video quality.
```
youtube-dl -f mp4/best -k -x https://youtu.be/uZOcBRa2YwI
```
If there's an issue downloading YouTube video, try:
```
brew update
brew upgrade youtube-dl
```

## Evaluating with a Pretrained Model
### Build a virtual environment and install packages
```
git clone https://github.com/ChunTao1999/cav-mae.git
python3 -m venv venv
source venv/bin/activate
TMPDIR=/var/tmp pip3 install -r requirements_a5.txt --extra-index-url https://download.pytorch.org/whl/cu116 
```

## Contact
If you have a question, please bring up an issue (preferred) or send me an email [tao88@purdue.edu].
