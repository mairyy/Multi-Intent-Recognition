from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset']

class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_feats, video_feats, audio_feats, comet_feats, sbert_feats):
        
        self.label_ids = torch.tensor(label_ids)
        self.text_feats = torch.tensor(text_feats)
        self.video_feats = torch.tensor(np.array(video_feats))
        self.audio_feats = torch.tensor(np.array(audio_feats))
        self.size = len(self.text_feats)
        self.comet_xReact = torch.tensor(comet_feats['xReact'])
        self.comet_xWant = torch.tensor(comet_feats['xWant'])
        self.sbert_xReact = torch.tensor(sbert_feats['xReact'])
        self.sbert_xWant = torch.tensor(sbert_feats['xWant'])

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': self.label_ids[index], 
            'text_feats': self.text_feats[index],
            'video_feats': self.video_feats[index],
            'audio_feats': self.audio_feats[index],
            'relation_feats': {
                'comet': {
                    'xReact': self.comet_xReact[index],
                    'xWant': self.comet_xWant[index]
                },
                'sbert': {
                    'xReact': self.sbert_xReact[index],
                    'xWant': self.sbert_xWant[index]
                }
            }
        } 
        return sample