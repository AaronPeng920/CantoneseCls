"""
Zhengwei Peng, 2024/2/15

This code is used to split all audio files in a folder or folder into several segments of the same length, 
discarding the last one that is less than the specified length.

USAGE: 
    python3 processing.py -i `file or dir path need to be processed` -o `save dir path` -c `config path`
"""

import os
import torchaudio
import argparse
import yaml

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help='A dir or file need to be processed.'
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help='A dir to save processed files.'
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help='A config file to process audios.'
    )
    
    opt = parser.parse_args()
    return opt
    
def split_audio(input, output, segment_duration=30, sr=44100):
    segment_samples = int(sr * segment_duration)
    input_files = []
    file_count = 0
    segment_count = 0
    if os.path.isdir(input):
        for file in os.listdir(input):
            if file[0] == '.':
                continue
            input_files.append(os.path.join(input, file))
    elif os.path.isfile(input):
        input_files.append(input)
    else:
        raise ValueError('input should be a file or dir.')
    
    if not os.path.isdir(output):
        raise ValueError('output should be a dir.')
        
    for input_file in input_files:
        waveform, sample_rate = torchaudio.load(input_file)
        total_segments = waveform.size(1) // segment_samples


        for i in range(total_segments):
            segment = waveform[:, i * segment_samples : (i + 1) * segment_samples]
            output_file = os.path.join(output, f"{os.path.splitext(os.path.basename(input_file))[0]}_{i}.wav")
            torchaudio.save(output_file, segment, sr)
            segment_count += 1
            
        file_count += 1
    
    print("本次处理了 {} 个音频文件, 产生 {} 个 {}s 的片段.".format(file_count, segment_count, segment_duration))
    

if __name__ == "__main__":
    opt = get_opt()
    with open(opt.config, 'r') as confread:
        try:
            config = yaml.safe_load(confread)['data']
        except yaml.YAMLError as e:
            print(e)
            
    split_audio(opt.input, opt.output, segment_duration=config['duration'], sr=config['sr'])
