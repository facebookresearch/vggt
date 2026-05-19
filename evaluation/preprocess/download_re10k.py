"""
References:
[cashiwamochi/RealEstate10K_Downloader](https://github.com/cashiwamochi/RealEstate10K_Downloader/blob/master/generate_dataset.py)
The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

datasets/sequences/re10k_test_1800.txt: test sequences chosen by [PoseDiffusion](https://github.com/facebookresearch/PoseDiffusion/blob/main/pose_diffusion/datasets/re10k_test_1800.txt)
However, some of the youtube videos are not available now, so we evaluate [Pi3](https://github.com/yyfz/Pi3) on datasets/sequences/re10k_test_1719.txt

You may run into 403 error when downloading youtube videos, please refer to original pytube/pytubefix repo for help or use other downloader like yt-dlp.
However, this script works for us when doing evaluation for [Pi3](https://github.com/yyfz/Pi3).

For resolutions, most sequences are (640, 360), with a few exceptions:
3b0b55657925fb34: (640, 272)
3e034bde9426ae9f: (640, 338)
2c2cfc0ac780a3aa: (640, 338)
"""

import os
import os.path as osp
import glob

from pytubefix import YouTube

class Data:
    def __init__(self, url, seqname, list_timestamps):
        self.url = url
        self.list_seqnames = []
        self.list_list_timestamps = []

        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def add(self, seqname, list_timestamps):
        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def __len__(self):
        return len(self.list_seqnames)


def process(data, seq_id, videoname, output_root):
    seqname = data.list_seqnames[seq_id]
    image_dir = os.path.join(output_root, seqname, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    else:
        print("[INFO] Something Wrong, stop process")
        return True

    list_str_timestamps = []
    for timestamp in data.list_list_timestamps[seq_id]:
        timestamp = int(timestamp/1000) 
        str_hour = str(int(timestamp/3600000)).zfill(2)
        str_min = str(int(int(timestamp%3600000)/60000)).zfill(2)
        str_sec = str(int(int(int(timestamp%3600000)%60000)/1000)).zfill(2)
        str_mill = str(int(int(int(timestamp%3600000)%60000)%1000)).zfill(3)
        _str_timestamp = str_hour+":"+str_min+":"+str_sec+"."+str_mill
        list_str_timestamps.append(_str_timestamp)

    # extract frames from a video
    for idx, str_timestamp in enumerate(list_str_timestamps):
        command = 'ffmpeg -ss '+str_timestamp+' -i '+videoname+' -vframes 1 -f image2 '+image_dir+'/'+str(data.list_list_timestamps[seq_id][idx])+'.png'
        # print("current command is {}".format(command))
        os.system(command)

    png_files = sorted(
        glob.glob(os.path.join(image_dir, "*.png")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    for idx, old_path in enumerate(png_files):
        new_name = f"{idx:04d}.png"
        new_path = os.path.join(image_dir, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {os.path.basename(old_path)} -> {new_name}")

    return False

def wrap_process(list_args):
    return process(*list_args)

class DataDownloader:
    def __init__ (
            self,
            meta_root: str,
            output_root: str,
            sequence_list: list,  # end with .txt
            mode: str = "test",
        ):
        print("[Re10k Downloader] Loading data list ... ")
        self.meta_root = meta_root
        all_seqnames = glob.glob(osp.join(meta_root, mode, '*.txt'))
        all_seqnames = sorted([osp.basename(x) for x in all_seqnames])
        all_seqnames = set(all_seqnames)

        the_other_mode = "train" if mode == "test" else "test"
        assert mode == "test", "Currently only support test mode, please set mode to 'test'"
        all_seq_exists = True
        seq_not_exists = []
        all_other_seqnames = {}
        for seqname in sequence_list:
            if seqname not in all_seqnames:
                if all_seq_exists:
                    all_other_seqnames = sorted(glob.glob(osp.join(meta_root, the_other_mode, '*.txt')))
                    all_other_seqnames = set(all_other_seqnames)
                
                if seqname not in all_other_seqnames:
                    print(f"[Error] {seqname} not in bote train and test meta")
                else:
                    print(f"[Warning] {seqname} not in {mode} meta, but in {the_other_mode} meta")
                seq_not_exists.append(seqname)
                all_seq_exists = False
        if not all_seq_exists:
            print("---------------------------------------------")
            print(seq_not_exists)
            raise ValueError(f"{mode} meta not exists, please check the path")
        print(f"[Re10k Downloader] {len(sequence_list)} sequences are to download in {mode} mode")

        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)
        self.mode = mode
        # self.sequence_list = sequence_list

        self.isDone = False

        self.list_data = []
        for txt_file in sequence_list:
            seq_name = txt_file.split('.')[0]
            if osp.exists(osp.join(output_root, seq_name)):
                print(f"[Re10k Downloader] {seq_name} already exists, skip")
                continue

            # extract info from txt
            txt_path = osp.join(self.meta_root, self.mode, txt_file)
            with open(txt_path, "r") as seq_file:
                lines = seq_file.readlines()
                youtube_url = ""
                list_timestamps= []
                for idx, line in enumerate(lines):
                    if idx == 0:
                        youtube_url = line.strip()
                    else:
                        timestamp = int(line.split(' ')[0])
                        list_timestamps.append(timestamp)

            isRegistered = False
            for i in range(len(self.list_data)):
                if youtube_url == self.list_data[i].url:
                    isRegistered = True
                    self.list_data[i].add(seq_name, list_timestamps)
                else:
                    pass

            if not isRegistered:
                self.list_data.append(Data(youtube_url, seq_name, list_timestamps))

        print("[Re10k Downloader] {} movies are used in {} mode".format(len(self.list_data), self.mode))


    def Run(self, tmp_dir):
        print("[Re10k Downloader] Start downloading {} movies".format(len(self.list_data)))

        os.makedirs(tmp_dir, exist_ok=True)
        for global_count, data in enumerate(self.list_data):
            print("[Re10k Downloader] Downloading {} ".format(data.url))
            try :
                # sometimes this fails because of known issues of pytube and unknown factors
                yt = YouTube(data.url)
                stream = yt.streams.first()
                stream.download(tmp_dir, data.url.split("=")[-1])
            except :
                failure_log = open(osp.join(self.output_root, 'failed_videos.txt'), 'a')
                for seqname in data.list_seqnames:
                    failure_log.writelines(seqname + '\n')
                failure_log.close()
                continue

            videoname = osp.join(tmp_dir, data.url.split("=")[-1])
            if len(data) == 1: # len(data) is len(data.list_seqnames)
                process(data, 0, videoname, self.output_root)
            else:
                for seq_id in range(len(data)):
                    process(data, seq_id, videoname, self.output_root)
                    print("Process {} done".format(seq_id))

            # remove videos
            command = "rm " + videoname 
            os.system(command)

            if self.isDone:
                return False

        return True

    def Show(self):
        print("########################################")
        global_count = 0
        for data in self.list_data:
            print(" URL : {}".format(data.url))
            for idx in range(len(data)):
                print(" SEQ_{} : {}".format(idx, data.list_seqnames[idx]))
                print(" LEN_{} : {}".format(idx, len(data.list_list_timestamps[idx])))
                global_count = global_count + 1
            print("----------------------------------------")

        print("TOTAL : {} sequnces".format(global_count))

if __name__ == "__main__":
    # setup_debug(True, 10033)
    MODE = "test"
    RE10K_METAROOT = "data/re10k/metadata"
    OUTPUT_ROOT = "data/re10k"
    SEQUENCE_LIST_FILE = "evaluation/preprocess/re10k_test_1800.txt"
    TMP_DIR = osp.join(OUTPUT_ROOT, "tmp")
    
    with open(SEQUENCE_LIST_FILE, "r") as f:
        sequence_list = f.read().splitlines()
        for idx, seq in enumerate(sequence_list):
            sequence_list[idx] = seq + '.txt' if seq[-4:] != '.txt' else seq

    downloader = DataDownloader(
        meta_root     = RE10K_METAROOT,
        output_root   = OUTPUT_ROOT,
        sequence_list = sequence_list,
        mode          = MODE,
    )

    downloader.Show()
    isOK = downloader.Run(tmp_dir=TMP_DIR)

    if isOK:
        print("Done!")
    else:
        print("Failed")