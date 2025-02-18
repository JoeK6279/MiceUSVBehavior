import argparse
import sys
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from audiomentations import Compose, TimeStretch, PitchShift, Shift
import seaborn as sns


class AudioDataset(Dataset):
    def __init__(self, files, labels, sample_rate=250000, duration=0.5, augment=False, normalize=False, hop_length=1024, n_fft=4096):
        self.files = files
        self.labels = labels
        self.sample_rate = sample_rate
        self.target_length = int(duration * sample_rate)
        self.augment = augment
        self.normalize=normalize
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=128,
            f_min=35000,
            f_max=125000
        )
        
        if self.augment:
            self.augment = Compose([
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                Shift(p=0.5),
            ])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(str(self.files[idx]))
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[1] > self.target_length:
            start = torch.randint(0, waveform.shape[1] - self.target_length, (1,))
            waveform = waveform[:, start:start + self.target_length]
        else:
            padding = self.target_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))

        
        if self.augment:
            waveform = torch.from_numpy(self.augment(waveform.numpy(), sample_rate=self.sample_rate))
        
    
        mel_spec = self.mel_transform(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(abs(mel_spec))

        if self.normalize:
            mel_spec = torchvision.transforms.Normalize(mean=[-51.7458], std=[7.2571])(mel_spec)
            
        
        return mel_spec, self.labels[idx]


class SimpleAudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_data(data_dir, seed=42):
    files = []
    labels = []
    
    class_files_dict = {}
    for class_idx, class_dir in enumerate(sorted(Path(data_dir).glob('*'))):
        class_files = list(class_dir.glob('*.wav'))
        class_files_dict[class_dir] = class_files

    files = []
    labels = []
    
    idx2class = {0: 'non_social', 1: 'social'}
    for count, (class_dir, class_files) in enumerate(class_files_dict.items()):
        files.extend(class_files)
        if str(class_dir).split('/')[-1] in ['exploring',"rearing","self_grooming","digging"]:
            labels.extend([0] * len(class_files))
        else:
            labels.extend([1] * len(class_files))
    
    return (files, labels), idx2class

def create_weighted_sampler(labels):
    class_counts = Counter(labels)
    total = len(labels) 
    
    class_weights = {class_idx: total / count for class_idx, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * correct / total
    val_loss = val_loss / len(dataloader)
    
    # Calculate per-class accuracy
    class_correct = {}
    class_total = {}
    for pred, label in zip(all_preds, all_labels):
        if label not in class_total:
            class_total[label] = 0
            class_correct[label] = 0
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    class_accuracies = {
        class_idx: (class_correct[class_idx] / class_total[class_idx]) * 100
        for class_idx in class_total
    }

    conf_matrix = confusion_matrix(all_labels, all_preds ,normalize='true')

    f1 = f1_score(all_labels, all_preds, average='weighted')

    
    return val_loss, accuracy, class_accuracies, conf_matrix, f1

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, scheduler=None, early_stopping_patience=10, name="", idx2class=None):
    best_val_loss = float('inf')
    best_f1 = 0
    patience_counter = 0
    training_history = []
    best_avg_accu=0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()*labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({'loss': train_loss/total,'acc': 100.*correct/total})

        val_loss, val_accuracy, class_accuracies, conf_matrix, f1 = validate(model, val_loader, 
                                                          criterion, device)
        
        avg_accu = np.array(list(class_accuracies.values())).mean()

        if scheduler is not None:
            scheduler.step(val_loss)
        
        torch.save({'state_dict':model.state_dict(),'epoch':epoch}, f'{name}_latest.pth')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), f'{name}_best_model.pth')
        else:
            patience_counter += 1
        
        if best_avg_accu < avg_accu:
            best_avg_accu = avg_accu
            torch.save({'state_dict':model.state_dict(),'epoch':epoch, 'avg_accu':avg_accu}, f'{name}_best_avg_accu_model.pth')
            
        import pandas as pd
        pd.options.display.float_format = '{:.1f}'.format
        cmtx = pd.DataFrame(
            conf_matrix
        )

        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {100.*correct/total:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Avg Acc: {avg_accu:.2f}, Val F1: {f1:.2f}')
        print(cmtx*100)
        plt.figure()
        sns.heatmap(cmtx, annot=True, fmt=".1%", cmap='coolwarm',xticklabels=[idx2class[n] for n in cmtx.columns], yticklabels=[idx2class[n] for n in cmtx.columns])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        fig = plt.gcf()
        fig.autofmt_xdate(rotation=45)
        ax = plt.gca()
        ax.tick_params(axis='x', rotation=45)
        plt.title(name)
        plt.savefig(f'{name}_confusion_matrix.png',bbox_inches='tight')
        plt.close()

        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss/len(train_loader)/inputs.shape[0],
            'train_acc': 100.*correct/total,
            'val_loss': val_loss,
            'val_acc': val_accuracy,
            'class_accuracies': class_accuracies
        })
        
        if patience_counter >= early_stopping_patience and epoch > 300:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    return pd.DataFrame(training_history)

def main():

    import torch
    torch.manual_seed(42)
    argp = argparse.ArgumentParser()
    argp.add_argument("--type", type=str, default="")
    argp.add_argument("--datapath", type=str, default="")
    argp.add_argument("--duration", type=str, default="05")
    argp.add_argument("--bs", type=int, default=64)
    argp.add_argument("--TEST", action='store_true')
    args = argp.parse_args(sys.argv[1:])


    # Hyperparameters
    batch_size = args.bs
    learning_rate = 0.0005
    num_epochs = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = f"mousr_{args.duration}_{args.type}"
    duration = 0.5 if args.duration == '05' else  1.5 if args.duration == '15' else int(args.duration)

    
    train_data, idx2class = create_data(f'{args.datapath}/mouse_{args.duration}s_train_{args.type}/')
    test_data, idx2class = create_data(f'{args.datapath}/mouse_{args.duration}s_test_{args.type}/')

    train_dataset = AudioDataset(train_data[0], train_data[1], augment=True, duration=duration, normalize=True, hop_length=1024, n_fft=4096)
    test_dataset = AudioDataset(test_data[0], test_data[1], augment=False, duration=duration, normalize=True, hop_length=1024, n_fft=4096)
    
    train_sampler = create_weighted_sampler(train_data[1])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = SimpleAudioClassifier(len(idx2class.keys())).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    if not args.TEST:
        history = train_model(
            model, train_loader, test_loader, criterion, optimizer,
            num_epochs, device, scheduler, early_stopping_patience=10, name=name, idx2class=idx2class
        )

    model.load_state_dict(torch.load(f'{name}_best_avg_accu_model.pth', weights_only=False)['state_dict'])
    test_loss, test_accuracy, test_class_accuracies, conf_matrix, f1 = validate(
        model, test_loader, criterion, device
    )
    import pandas as pd
    pd.options.display.float_format = '{:.1f}'.format
    cmtx = pd.DataFrame(
        conf_matrix
    )
    avg_accu = np.array(list(test_class_accuracies.values())).mean()
    print(f'{name}= {np.array(conf_matrix).tolist()}')
    
    plt.figure()
    sns.heatmap(cmtx, annot=True, fmt=".1%", cmap='coolwarm',xticklabels=[idx2class[n] for n in cmtx.columns], yticklabels=[idx2class[n] for n in cmtx.columns])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fig = plt.gcf()
    fig.autofmt_xdate(rotation=45)
    ax = plt.gca()
    ax.tick_params(axis='x', rotation=45)
    plt.title(name)
    plt.savefig(f'{name}_confusion_matrix.png',bbox_inches='tight')
    plt.close()

main()