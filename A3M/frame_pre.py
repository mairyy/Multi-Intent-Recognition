#Data contains frames cut by MintRec's author. Download at: https://drive.google.com/drive/folders/1_PNRQRsdvUSmbCyKigkr5MknLIKAK58e
import os
import pandas as pd
from pathlib import Path

def select_frame(src, des, text_path):
    """
    Select a frame corresponding to each utterance. 
    Return:
        Frames which their name matches with format: season_episode_clip_* (ex: S04_E01_29_0_4.jpg)
    """
    if not os.path.exists(des):
        os.mkdir(des)

    map_path = des + '_text_image.csv'

    print('Beginning selecting frames...')
    data = pd.read_csv(text_path, sep='\t')
    frames_path = []
    for i in range(len(data)):
        file_name = str(data['season'][i]) + '_' + str(data['episode'][i]) + '_' + str(data['clip'][i])
        file = ''
        for f in os.listdir(src):
            if str(f).startswith(file_name):
                file = str(f)
                os.rename(src + '/' + file, des + '/' + file_name + '.jpg')
                break
        if file == '':
            print("Cannot find frame starting with " + file_name)
        frames_path.append(file)
    
    log = pd.DataFrame(data[['season', 'episode', 'clip']])
    log['frame_name'] = frames_path
    log.to_csv(map_path)
    print("Done! Selected frames were saved at " + des)

if __name__ == '__main__':
    if not os.path.exists('../datasets/MIntRec/selected_frames'):
        os.makedirs('../datasets/MIntRec/selected_frames')
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    for f in files:
        select_frame('../datasets/MIntRec/raw_frames', '../datasets/MIntRec/selected_frames' + '/' + f[:len(f)-4], '../datasets/MIntRec/' + f)