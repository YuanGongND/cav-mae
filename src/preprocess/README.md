## Data Preparation

You can of course use the `CAV-MAE` model with your own training pipeline, but the input still needs to be identical to ours if you want to use our pretrained model. 

Otherwise, if you want to use our pipeline (including any of pretraining, classification, retrieval, and inpainting). You will need to prepare the data in the same format as us. 

### Step 1. Extract audio track and image frames from the video

Suppose you have a set of videos (e.g., in `.mp4` format), you will need to first extract the audio track and image frames offline and save them on the disk, doing it on-the-fly usually dramatically increases the data loading overhead. In `src/preprocess/extract_{audio,video_frame}.py`, we include our code to do the extraction. Both scripts are simple, you will need to prepare a `csv` file containing a list of video paths (see `src/preprocess/sample_video_extract_list.csv` for an example) and `target_fold` (a single path) of your desired place to save the output. 

By default, we assume the `video_id` is the name of the video without extension and path, e.g., the `video_id` of `/path/test12345.mp4` is `test12345`. the output image frames will be saved at `target_fold/frame_{0-9}/video_id.jpg`, the output audio track will be saved at `target_fold/video_id.wav`.

The audio and image `target_fold` is better to be different, please record the `target_fold` for the next step. 

We provide a minimal example. The video and list are provided in this repo, you can just run to generate the frames and audio:
```python
cd cav-mae/src/preprocess
# extract video frames
python extract_video_frame.py -input_file_list sample_video_extract_list.csv -target_fold ./sample_frames
# extract audio tracks
python extract_audio.py  -input_file_list sample_video_extract_list.csv -target_fold ./sample_audio
```

### Step 2. Build a label set and json file for your dataset.

You will need two files:

- A label csv file listing all labels. (see `src/preprocess/sample_datafiles/class_labels_indices_as.csv` as an example).
- A json file that have four keys for each sample (see `src/preprocess/sample_datafiles/sample_json_as.json` as an example):
  - `wav`: the absolute path to the audio track extracted in the previous step, e.g., `/data/sls/audioset/--4gqARaEJE.flac`
  - `video_id`: the video_id (i.e., the video filename without extension), e.g., `--4gqARaEJE` for video `--4gqARaEJE.mp4`.
  - `video_path`: the `target_fold` you used in the previous step, e.g., `/data/sls/audioset/`. Our pipeline will load from `video_path/frame_{0-9}/video_id.jpg`, not `video_path/video_id.jpg` So **make sure `video_path/frame_{0-9}/video_id.jpg` contains your image frames.**
  - `labels`: all labels of this sample, if more than one, use `,` to separate, must be consistent with the label csv file.
  - You can see how we automatically generate such json with `src/preprocess/create_json_as.py`.

To make this easier, we share our AudioSet and VGGSound datafiles at [here](#audioset-and-vggsound-data-lists), you can use/modify based on our files. The shared datafiles also show the exact sample ids we used for our experiments, which may be helpful for reproduction purposes.
