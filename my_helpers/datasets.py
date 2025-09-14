import random

import torch
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data import Sampler


class ShuffledTrialSampler(Sampler):
    def __init__(self, dataset, trial_ids):
        super().__init__()
        self.indices = [i for i, (trial_num, *_rest) in enumerate(dataset.samples)
                        if trial_num in trial_ids]

    def __iter__(self):
        shuffled = self.indices.copy()
        random.shuffle(shuffled)
        return iter(shuffled)

    def __len__(self):
        return len(self.indices)


class TrialSampler(Sampler):
    """
    DataLoader(dataset, batch_size=32, sampler=TrialSampler(dataset, train_trials))
    """
    def __init__(self, dataset, trial_ids):
        super().__init__()
        self.indices = [i for i, (trial_num, *_rest) in enumerate(dataset.samples) if trial_num in trial_ids]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class AudioDataset(Dataset):
    def __init__(self, tools, behaviors, objects, obj_label_map, data_dict):
        self.samples = []
        self.tool2id = {t: i for i, t in enumerate(tools)}
        self.beh2id = {b: i for i, b in enumerate(behaviors)}
        self.obj2id = obj_label_map

        for beh in behaviors:
            for tool in tools:
                for obj in objects:
                    trials = data_dict[beh][tool][obj]
                    for i, trial in enumerate(trials):
                        self.samples.append((
                            i, trial,  # audio embedding data
                            self.obj2id[obj],
                            self.tool2id[tool],
                            self.beh2id[beh],
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trial_num, audio_embed, obj_id, tool_id, beh_id = self.samples[idx]
        return {
            "trial_num": torch.tensor(trial_num, dtype=torch.long),
            "audio_data": torch.tensor(audio_embed, dtype=torch.float32),
            "obj_id": torch.tensor(obj_id, dtype=torch.long),
            "tool_id": torch.tensor(tool_id, dtype=torch.long),
            "beh_id": torch.tensor(beh_id, dtype=torch.long),
        }


class AudioTextDataset(Dataset):
    def __init__(self, tools, behaviors, objects, obj_label_map, data_dict, text_dict):
        self.samples = []
        self.tool2id = {t: i for i, t in enumerate(tools)}
        self.beh2id = {b: i for i, b in enumerate(behaviors)}
        self.obj2id = obj_label_map

        for beh in behaviors:
            for tool in tools:
                for obj in objects:
                    text_emb = text_dict[beh][tool][obj]
                    trials = data_dict[beh][tool][obj]
                    for i, trial in enumerate(trials):
                        self.samples.append((
                            i, trial,  # audio embedding data
                            text_emb,  # same dim as trial
                            self.obj2id[obj],
                            self.tool2id[tool],
                            self.beh2id[beh]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trial_num, audio_embed, text_emb, obj_id, tool_id, beh_id = self.samples[idx]
        return {
            "trial_num": torch.tensor(trial_num, dtype=torch.long),
            "audio_data": torch.tensor(audio_embed, dtype=torch.float32),
            "text_data": torch.tensor(text_emb, dtype=torch.float32),
            "obj_id": torch.tensor(obj_id, dtype=torch.long),
            "tool_id": torch.tensor(tool_id, dtype=torch.long),
            "beh_id": torch.tensor(beh_id, dtype=torch.long),
        }
