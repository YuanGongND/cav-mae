import re
import math
from subprocess import check_call, PIPE, Popen
import shlex

re_metadata = re.compile('Duration: (\d{2}):(\d{2}):(\d{2})\.\d+,.*\n.* (\d+(\.\d+)?) fps')

def get_metadata(filename):
    '''
    Get video metadata using ffmpeg
    '''
    p1 = Popen(["ffmpeg", "-hide_banner", "-i", filename], stderr=PIPE, universal_newlines=True)
    output = p1.communicate()[1]
    matches = re_metadata.search(output)
    if matches:
        video_length = int(matches.group(1)) * 3600 + int(matches.group(2)) * 60 + int(matches.group(3))
        video_fps = float(matches.group(4))
        # print('video_length = {}\nvideo_fps = {}'.format(video_length, video_fps))
    else:
        raise Exception("Can't parse required metadata")
    return video_length, video_fps


def split_cut(filename, n, by='size'):
    '''
    Split video by cutting and re-encoding: accurate but very slow
    Adding "-c copy" speed up the process but causes imprecise chunk durations
    Reference: https://stackoverflow.com/a/28884437/1862500
    '''
    assert n > 0
    assert by in ['size', 'count']
    split_size = n if by == 'size' else None
    split_count = n if by == 'count' else None
    
    # parse meta data
    video_length, video_fps = get_metadata(filename)

    # calculate split_count
    if split_size:
        split_count = math.ceil(video_length / split_size)
        if split_count == 1:        
            raise Exception("Video length is less than the target split_size.")    
    else: #split_count
        split_size = round(video_length / split_count)

    output = []
    for i in range(split_count):
        split_start = split_size * i
        pth, ext = filename.rsplit(".", 1)
        output_path = '{}-{}.{}'.format(pth, i+1, ext)
        cmd = 'ffmpeg -hide_banner -loglevel panic -ss {} -t {} -i "{}" -y "{}"'.format(
            split_start, 
            split_size, 
            filename, 
            output_path
        )
        # print(cmd)
        check_call(shlex.split(cmd), universal_newlines=True)
        output.append(output_path)
    return output


if __name__ == '__main__':
    input_file_list = a
    input_filelist = np.loadtxt(input_file_list, delimiter=',', dtype=str, usecols=0)
    # outputs = split_cut(filename=)