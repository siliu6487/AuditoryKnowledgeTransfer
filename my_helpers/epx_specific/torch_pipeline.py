import copy
import os
import pickle

import numpy as np
import torch
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader

import configs
from my_helpers.datasets import AudioDataset, AudioTextDataset


def has_key_X(d):
    if isinstance(d, dict):
        # If it's the last layer and contains 'X'
        if "X" in d:
            return True
        # Otherwise, recurse into nested dicts
        return any(has_key_X(v) for v in d.values())
    return False


class TransferPipeline:
    def __init__(self, transfer_type: str, context_dict: dict, modalities: list,
                 audio_data_name: str, text_data_name: str or None = None,
                 tool_data_name: str or None = None,
                 other_modality_dict: dict or None = None,
                 data_path="/Users/siliu/Desktop/KnowledgeTransferMini/data"):
        self.device = configs.device
        self.transfer_type = transfer_type  # sincere or CLAP
        self.context_dict = context_dict

        assert set(self.context_dict['shared_object_list']).isdisjoint(self.context_dict['test_object_list'])
        self.full_obj_list = sorted(self.context_dict['shared_object_list'] + self.context_dict['test_object_list'])
        self.full_beh_list = sorted(list(set(self.context_dict['source_beh_list']
                                             + self.context_dict['target_beh_list'])))
        self.full_tool_list = sorted(list(set(self.context_dict['source_tool_list']
                                              + self.context_dict['target_tool_list'])))

        self.modalities = modalities
        self.data_path = data_path

        # ---------- required data ----------
        self.audio_data_name = audio_data_name  # saved interaction audio embeddings from preprocessing or pre-trained model
        self.audio_data_dict = self.get_data_dict(self.audio_data_name)
        self.audio_data_dict = self.get_unimodal_dict(self.audio_data_dict, modality="audio")

        # # ---------- CLAP transfer ----------
        self.text_data_dict = None if text_data_name is None else self.get_data_dict(text_data_name)

        # # ---------- optional experiments ----------
        # # tool-explore audio embeddings from preprocessing or pre-trained model
        self.tool_data_name = tool_data_name
        if self.tool_data_name is not None:
            self.tool_data_dict = self.get_data_dict(self.tool_data_name)
        #
        # self.other_modality_dict = other_modality_dict

    def get_data_dict(self, data_name):
        data_folder = os.path.join(self.data_path, data_name)
        data_dict = {}
        if ".pickle" in data_name:
            pass  # todo: load pickle
        elif ".bin" in data_name:
            bin_file = open(data_folder, 'rb')
            data_dict = pickle.load(bin_file)
            bin_file.close()
        elif ".npz" in data_name:
            with np.load(data_folder, allow_pickle=True) as loaded:
                data_dict = {k: loaded[k].item() if loaded[k].ndim == 0 else loaded[k]
                             for k in loaded.files}
        else:
            raise Exception(f"data_name not eligible: {data_name}")
        return data_dict

    def get_unimodal_dict(self, orig_dict, modality):
        if has_key_X(orig_dict):
            new_dict = {
                beh: {
                    tool: {
                        obj: [] for obj in self.full_obj_list}
                    for tool in self.full_tool_list
                }
                for beh in self.full_beh_list
            }
            for beh in new_dict.keys():
                for tool in new_dict[beh].keys():
                    for obj in new_dict[beh][tool].keys():
                        new_dict[beh][tool][obj] = orig_dict[beh][tool][modality][obj]['X']
            return new_dict
        else:
            return orig_dict

    def get_audio_text_dataset(self, tools, behaviors, objects, obj_label_map):
        return AudioTextDataset(
            tools=tools,
            behaviors=behaviors,
            objects=objects,
            obj_label_map=obj_label_map,
            data_dict=self.audio_data_dict,
            text_dict=self.text_data_dict,
        )

    def get_audio_dataset(self, tools, behaviors, objects, obj_label_map):
        return AudioDataset(
            tools=tools,
            behaviors=behaviors,
            objects=objects,
            obj_label_map=obj_label_map,
            data_dict=self.audio_data_dict
        )
