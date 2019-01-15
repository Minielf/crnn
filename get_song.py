import collections
import requests
import os
import time
import sys

# YUHANG
# PROJECT_PATH = "/Users/yuhangyang/Dropbox (Brown)/PROJECT/"
# START_INDEX = 23000

# ZEMIAO
# PROJECT_PATH = "/Users/zhuzem/Dropbox (Brown)/PROJECT/"
# START_INDEX = 23000

# MIAOLIN
# PROJECT_PATH = "/Users/mindy927/Dropbox/PROJECT/"
# START_INDEX = 22905

SLEEP_TIME = 5
COOLDOWN_TIME = 200
SEGMENT_SIZE = 1000

def process_info(file_name):
    infos = collections.defaultdict(list)
    with open(file_name, "r") as r:
        for line in r:
            msd_id = line[:18]
            i = 46
            singer = ""
            while i < len(line):
                if line[i] == '<':
                    break
                singer += line[i]
                i += 1
            singer = singer.replace(' ', '+')
            i += 5
            song = line[i:-1]
            song = song.replace(' ', '+')
            infos[msd_id] = [singer, song]
    return infos

def download_url(start_index, id_filename = 'test_x_msd_id.txt'):
    infos = process_info('unique_tracks.txt')
    ids = []
    with open(id_filename) as f:
        ids = f.read().split('\n')

    for index in range(start_index, len(ids)):
        try:
            id = ids[index]
            txt_name = id + '.txt'
            text_path = './TRAIN_TEXT_FILE/TRAIN_TEXT_FILE_2/' + txt_name
            audio_name = id + '.m4a'
            audio_path = './TRAIN_AUDIO_DATA/TRAIN_AUDIO_DATA_2/' + audio_name

            audio_exist = os.path.isfile(audio_path)
            if audio_exist:
                continue

            time.sleep(SLEEP_TIME)

            url = "https://itunes.apple.com/search?term=" + str(infos[id][0]) + "+" + str(infos[id][1])
            url_r = requests.get(url, allow_redirects=True, stream=False)
            open(text_path, 'wb').write(url_r.content)

            JSONcontent = url_r.content.decode()
            audio_url = ""
            if JSONcontent.find("previewUrl") != -1:
                i = JSONcontent.find("previewUrl") + len("previewUrl") + 3
                while i < len(JSONcontent):
                    if JSONcontent[i] == '"':
                        break
                    audio_url += JSONcontent[i]
                    i += 1

                audio_r = requests.get(audio_url, allow_redirects=True, stream=False)
                open(audio_path, 'wb').write(audio_r.content)

                print("Audio for id[", index, "]: " + id + " Downloaded with size ", len(audio_r.content))
            else:
                print("Audio for id[", index, "]: " + id + " Failed to download.")

        except requests.exceptions.ConnectionError:
            print("Crashed at time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            time.sleep(COOLDOWN_TIME)

    

if __name__ == '__main__':
    assert len(sys.argv) >= 2, "Required arguments missing: id_start_index"

    if len(sys.argv) >= 3:
        download_url(int(sys.argv[1]), sys.argv[2])
    else:
        download_url(int(sys.argv[1]))





