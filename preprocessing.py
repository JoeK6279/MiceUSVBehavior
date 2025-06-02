import numpy as np
import librosa
from scipy import signal
import torch
import argparse
import torchaudio
import os
import pandas as pd
import shutil

class UltrasonicPreprocessor:
    def __init__(
        self,
        sample_rate=250000,   
        min_freq=40000,       
        max_freq=125000,      
        segment_duration=0.5, 
        step = 0.05
    ):
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.segment_samples = int(segment_duration * sample_rate)
        self.step_samples = int(step * sample_rate)
        
    def compute_mel_spectrogram(self, audio):
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=250000,
            n_fft=4096,
            hop_length=1024,
            n_mels=128,
            fmin=self.min_freq,
            fmax=self.max_freq)
        return S

    def compute_stft_spectrogram(self, audio):
        S = librosa.stft(
            audio,
            n_fft=512,
            hop_length=64,
            win_length=512,
            window= signal.windows.kaiser(512, beta=5)
        )
        return abs(S)
    
    def detect_active_segments(self, audio):

        D = self.compute_stft_spectrogram(audio)

        D = D[int(self.min_freq*D.shape[0]/self.sample_rate):,:]

        active_segments = np.max(D,axis=0)>0.1
        if np.mean(active_segments)>0.1:
            return True
        return False
    
    def extract_segments(self, audio):
        segments = []
        segments_time = []
        start_idx = 0
        
        while start_idx + self.segment_samples <= len(audio):
            segment = audio[start_idx:start_idx + self.segment_samples]
            
            if self.detect_active_segments(segment):
                segments.append(segment)
                segments_time.append((start_idx, start_idx + self.segment_samples))
                start_idx += self.segment_samples//2
            else:
                start_idx += self.step_samples
        return segments, segments_time
    
    def process_file(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        return self.extract_segments(audio)


def main(d=0, datapath='', outpath = ''):
    d_text = d if d in [1,2]  else str(d).replace('.','')
    
    out = ''
    for f in os.listdir(f'{datapath}/WT 3 weeks'):
        preprocessor = UltrasonicPreprocessor(segment_duration=d)
        print(f"{datapath}/WT 3 weeks/{f}")
        segments, segments_time= preprocessor.process_file(f"{datapath}/WT 3 weeks/{f}")
        df = pd.read_excel(f"{datapath}/WT 3 weeks/WT 3 weeks NUOVI.xlsx", sheet_name=f.split('.')[0])
        cut_time = df['Time_Relative_sf'].to_numpy()
        labels = df['Behavior'].to_numpy()
        cut_time = cut_time[::2]*250000
        labels = labels[::2]
        labels = [x.replace(' ','_').replace('/','_').replace('(','_').replace(')','_').replace(',','_') for x in labels]

        out_labels = []
        for x in labels:
            if x =='nose-nose_sniffing':
                out_labels.append('nose_sniffing')
            elif x=='contact__crawl_over_under__allogrooming_':
                out_labels.append('contact')
            elif x=='rearing_wall_rearing':
                out_labels.append('rearing')
            else:
                out_labels.append(x)
        labels = out_labels

        count=0
        test_list = ['topo 1 VH WT 3 weeks.WAV','topo 7 VH WT 3 weeks.WAV','topo 13 WT 1 mg-kg 3 weeks.WAV','topo 19 WT 1 mg-kg 3 weeks.WAV','topo 25 WT 10 mg-kg 3 weeks.WAV',]
        if f in test_list:
            print(f)
            for n, (seg, seg_time) in enumerate(zip(segments, segments_time)):
                s_idx = np.argmax(cut_time>seg_time[0])-1
                os.makedirs(f'{outpath}/mouse_{d_text}s_test/{labels[s_idx]}',exist_ok=True)
                if  cut_time[s_idx+1]>=seg_time[1]: 
                    out+=f'{f},{count},{labels[s_idx]},{seg_time[0]},{seg_time[1]}\n'
                    torchaudio.save(f'{outpath}/mouse_{d_text}s_test/{labels[s_idx]}/WT_{f}_{count}.wav', torch.from_numpy(seg.copy()).unsqueeze(0).float(), 250000)
                    count+=1
            
        else:
            for n, (seg, seg_time) in enumerate(zip(segments, segments_time)):
                s_idx = np.argmax(cut_time>seg_time[0])-1
                os.makedirs(f'{outpath}/mouse_{d_text}s_train/{labels[s_idx]}',exist_ok=True)
                if  cut_time[s_idx+1]>=seg_time[1]:
                    out+=f'{f},{count},{labels[s_idx]},{seg_time[0]},{seg_time[1]}\n'
                    torchaudio.save(f'{outpath}/mouse_{d_text}s_train/{labels[s_idx]}/WT_{f}_{count}.wav', torch.from_numpy(seg.copy()).unsqueeze(0).float(), 250000)
                    count+=1
            
    open(f'{outpath}/mouse_{d_text}s_WT_3_weeks.txt','w').write(out)

    out = ''
    for f in range(1,9):
        preprocessor = UltrasonicPreprocessor(segment_duration=d)
        print(f"{datapath}/WT vh 6 weeks/T{f:07d}.WAV")
        segments, segments_time= preprocessor.process_file(f"{datapath}/WT vh 6 weeks/T{f:07d}.WAV")
        df = pd.read_excel(f"{datapath}/WT vh 6 weeks/topo {f} - 6 weeks.xlsx")
        cut_time = np.concatenate((np.array([0]), df['Unnamed: 0'].values[2:-2]))
        labels = df['Unnamed: 3'].values[2:-2]
        labels[::2] = 'exploring'
        labels = np.append(labels, 'exploring')
        cut_time = cut_time*250000
        labels = [x.lower().replace(' ','_').replace('/','_').replace('(','_').replace(')','_').replace(',','_') for x in labels]

        count=0
        if f==1 or f==7:
            for n, (seg, seg_time) in enumerate(zip(segments, segments_time)):
                s_idx = np.argmax(cut_time>seg_time[0])-1
                os.makedirs(f'{outpath}/mouse_{d_text}s_test/{labels[s_idx]}',exist_ok=True)
                if  cut_time[s_idx+1]>=seg_time[1]:
                    out+=f'{f},{count},{labels[s_idx]},{seg_time[0]},{seg_time[1]}\n'
                    torchaudio.save(f'{outpath}/mouse_{d_text}s_test/{labels[s_idx]}/WT_{f}_{count}.wav', torch.from_numpy(seg.copy()).unsqueeze(0).float(), 250000)
                    count+=1
        else:
            for n, (seg, seg_time) in enumerate(zip(segments, segments_time)):
                s_idx = np.argmax(cut_time>seg_time[0])-1
                os.makedirs(f'{outpath}/mouse_{d_text}s_train/{labels[s_idx]}',exist_ok=True)
                if  cut_time[s_idx+1]>=seg_time[1]: 
                    out+=f'{f},{count},{labels[s_idx]},{seg_time[0]},{seg_time[1]}\n'
                    torchaudio.save(f'{outpath}/mouse_{d_text}s_train/{labels[s_idx]}/WT_{f}_{count}.wav', torch.from_numpy(seg.copy()).unsqueeze(0).float(), 250000)
                    count+=1
    open(f'{outpath}/mouse_{d_text}s_WT_6_weeks.txt','w').write(out)


    out = ''
    for f in os.listdir(f'{datapath}/KO 3 weeks'):
        if not (f.endswith('.WAV') or f.endswith('.wav')):
            continue
        preprocessor = UltrasonicPreprocessor(segment_duration=d)
        print(f"{datapath}/KO 3 weeks/{f}")
        segments, segments_time= preprocessor.process_file(f"{datapath}/KO 3 weeks/{f}")
        df = pd.read_excel(f"{datapath}/KO 3 weeks/KO 3 weeks NUOVI.xlsx", sheet_name=f.split('.')[0])
        cut_time = df['Time_Relative_sf'].to_numpy()
        labels = df['Behavior'].to_numpy()
        cut_time = cut_time[::2]*250000
        labels = labels[::2]
        labels = [x.replace(' ','_').replace('/','_').replace('(','_').replace(')','_').replace(',','_') for x in labels]

        out_labels = []
        for x in labels:
            if x =='nose-nose_sniffing':
                out_labels.append('nose_sniffing')
            elif x=='contact__crawl_over_under__allogrooming_':
                out_labels.append('contact')
            elif x=='rearing_wall_rearing':
                out_labels.append('rearing')
            else:
                out_labels.append(x)
        labels = out_labels

        count=0
        test_list = ['topo 1 VH KO 3 weeks.WAV','topo 7 VH KO 3 weeks.WAV','topo 13 KO 1 mg-kg 3 weeks.WAV','topo 19 KO 1 mg-kg 3 weeks.WAV','topo 25 KO 10 mg-kg 3 weeks.WAV',]
        if f in test_list:
            for n, (seg, seg_time) in enumerate(zip(segments, segments_time)):
                s_idx = np.argmax(cut_time>seg_time[0])-1
                os.makedirs(f'{outpath}/mouse_{d_text}s_test/{labels[s_idx]}',exist_ok=True)
                if  cut_time[s_idx+1]>=seg_time[1]:
                    out+=f'{f},{count},{labels[s_idx]},{seg_time[0]},{seg_time[1]}\n'
                    torchaudio.save(f'{outpath}/mouse_{d_text}s_test/{labels[s_idx]}/KO_{f}_{count}.wav', torch.from_numpy(seg.copy()).unsqueeze(0).float(), 250000)
                    count+=1
            
        else:
            for n, (seg, seg_time) in enumerate(zip(segments, segments_time)):
                s_idx = np.argmax(cut_time>seg_time[0])-1
                os.makedirs(f'{outpath}/mouse_{d_text}s_train/{labels[s_idx]}',exist_ok=True)
                if  cut_time[s_idx+1]>=seg_time[1]: 
                    out+=f'{f},{count},{labels[s_idx]},{seg_time[0]},{seg_time[1]}\n'
                    torchaudio.save(f'{outpath}/mouse_{d_text}s_train/{labels[s_idx]}/KO_{f}_{count}.wav', torch.from_numpy(seg.copy()).unsqueeze(0).float(), 250000)
                    count+=1
    open(f'{outpath}/mouse_{d_text}s_KO_3_weeks.txt','w').write(out)


    out = ''
    for f in range(3,10):
        preprocessor = UltrasonicPreprocessor(segment_duration=d)
        print(f"{datapath}/KO vh 6 weeks/T{f:07d}.WAV")
        segments, segments_time= preprocessor.process_file(f"{datapath}/KO vh 6 weeks/T{f:07d}.WAV")
        df = pd.read_excel(f"{datapath}/KO vh 6 weeks/topo {f} - 6 weeks.xlsx")
        cut_time = np.concatenate((np.array([0]), df['Unnamed: 0'].values[2:-2]))
        labels = df['Unnamed: 3'].values[2:-2]
        labels[::2] = 'exploring'
        labels = np.append(labels, 'exploring')
        cut_time = cut_time*250000
        labels = [x.lower().replace(' ','_').replace('/','_').replace('(','_').replace(')','_').replace(',','_') for x in labels]

        count=0
        if f==3 or f==7:
            for n, (seg, seg_time) in enumerate(zip(segments, segments_time)):
                s_idx = np.argmax(cut_time>seg_time[0])-1
                os.makedirs(f'{outpath}/mouse_{d_text}s_test/{labels[s_idx]}',exist_ok=True)
                if  cut_time[s_idx+1]>=seg_time[1]: 
                    out+=f'{f},{count},{labels[s_idx]},{seg_time[0]},{seg_time[1]}\n'
                    torchaudio.save(f'{outpath}/mouse_{d_text}s_test/{labels[s_idx]}/KO_{f}_{count}.wav', torch.from_numpy(seg.copy()).unsqueeze(0).float(), 250000)
                    count+=1
        else:
            for n, (seg, seg_time) in enumerate(zip(segments, segments_time)):
                s_idx = np.argmax(cut_time>seg_time[0])-1
                os.makedirs(f'{outpath}/mouse_{d_text}s_train/{labels[s_idx]}',exist_ok=True)
                if  cut_time[s_idx+1]>=seg_time[1]:
                    out+=f'{f},{count},{labels[s_idx]},{seg_time[0]},{seg_time[1]}\n'
                    torchaudio.save(f'{outpath}/mouse_{d_text}s_train/{labels[s_idx]}/KO_{f}_{count}.wav', torch.from_numpy(seg.copy()).unsqueeze(0).float(), 250000)
                    count+=1
    open(f'{outpath}/mouse_{d_text}s_KO_6_weeks.txt','w').write(out)

    for i in os.listdir(f'{outpath}/mouse_{d_text}s_train'):
        if not i.endswith('.txt'):
            os.makedirs(f'{outpath}/mouse_{d_text}s_train_KO/{i}',exist_ok=True)
            os.makedirs(f'{outpath}/mouse_{d_text}s_train_WT/{i}',exist_ok=True)
            for j in os.listdir(f'{outpath}/mouse_{d_text}s_train/{i}'):
                if 'KO' in j:
                    os.system(f'cp "{outpath}/mouse_{d_text}s_train/{i}/{j}" "{outpath}/mouse_{d_text}s_train_KO/{i}/{j}"')
                else:
                    os.system(f'cp "{outpath}/mouse_{d_text}s_train/{i}/{j}" "{outpath}/mouse_{d_text}s_train_WT/{i}/{j}"')

    for i in os.listdir(f'{outpath}/mouse_{d_text}s_test'):
        if not i.endswith('.txt'):
            os.makedirs(f'{outpath}/mouse_{d_text}s_test_KO/{i}',exist_ok=True)
            os.makedirs(f'{outpath}/mouse_{d_text}s_test_WT/{i}',exist_ok=True)
            for j in os.listdir(f'{outpath}/mouse_{d_text}s_test/{i}'):
                if 'KO' in j:
                    os.system(f'cp "{outpath}/mouse_{d_text}s_test/{i}/{j}" "{outpath}/mouse_{d_text}s_test_KO/{i}/{j}"')
                else:
                    os.system(f'cp "{outpath}/mouse_{d_text}s_test/{i}/{j}" "{outpath}/mouse_{d_text}s_test_WT/{i}/{j}"')

    shutil.rmtree(f'{outpath}/mouse_{d_text}s_train')
    shutil.rmtree(f'{outpath}/mouse_{d_text}s_test')


pars = argparse.ArgumentParser()
import sys
pars.add_argument('--d', type=float, default=0.5)
pars.add_argument('--datapath', type=str, default='')
pars.add_argument('--outpath', type=str, default='')
args = pars.parse_args(sys.argv[1:])
main(args.d, args.datapath, args.outpath)