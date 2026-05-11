import copy
import os
import pickle

import numpy as np
import torch
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader

import configs
from my_helpers.datasets import AudioDataset, AudioTextDataset
from my_helpers.models import LinearProbLayer


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
        self.full_beh_list = sorted(list(set(self.context_dict['source_beh_list']
                                             + self.context_dict['target_beh_list'])))
        self.full_tool_list = sorted(list(set(self.context_dict['source_tool_list']
                                              + self.context_dict['target_tool_list'])))

        self.modalities = modalities
        self.data_path = data_path

        # ---------- required data ----------
        self.audio_data_name = audio_data_name  # saved interaction audio embeddings from preprocessing or pre-trained model
        self.audio_data_dict = self.get_data_dict(self.audio_data_name)
        self.audio_data_dict = self.get_unimodal_dict(self.audio_data_dict, modality="audio",
                                                      full_obj_dict=self.get_full_obj_list())

        # # ---------- CLAP transfer ----------
        self.text_data_dict = None if text_data_name is None else self.get_data_dict(text_data_name)

        # # ---------- optional experiments ----------
        # # tool-explore audio embeddings from preprocessing or pre-trained model
        self.tool_data_name = tool_data_name
        if self.tool_data_name is not None:
            self.tool_data_dict = self.get_data_dict(self.tool_data_name)
        #
        # self.other_modality_dict = other_modality_dict

    def get_full_obj_list(self):
        return sorted(list(set(self.context_dict['shared_object_list'] + self.context_dict['test_object_list'])))

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

    def get_unimodal_dict(self, orig_dict, modality, full_obj_dict):
        if has_key_X(orig_dict):
            new_dict = {
                beh: {
                    tool: {
                        obj: [] for obj in full_obj_dict}
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

    def train_linear_probe_clf(self, train_dataloader, val_dataloader, hyparams, obj_list, encoder):
        assert encoder is not None

        clf = LinearProbLayer(in_dim=hyparams['encoder_output_dim'], num_classes=len(obj_list)).to(self.device)
        optimizer = torch.optim.AdamW(clf.parameters(), lr=hyparams['lr_classifier'])
        loss_func = torch.nn.CrossEntropyLoss()
        encoder.eval()

        all_losses = []
        all_accuracies = []
        all_losses_val = []
        all_accuracies_val = []
        for epoch in range(hyparams['epoch_classifier']):
            total_correct = 0
            total_truth = 0
            loss_value = 0
            for batch in train_dataloader:
                audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
                with torch.no_grad():
                    audio_enc = encoder(audio_data)

                pred = clf(audio_enc).view(-1, len(obj_list))  # (num_data, num_class)
                truth = obj_id.view(-1).long()
                loss = loss_func(pred, truth)
                loss_value += loss.item()

                truth = truth.detach().cpu().numpy()
                pred_label = torch.argmax(pred, dim=-1).detach().cpu().numpy()
                correct_num = np.sum(pred_label == truth)
                total_correct += correct_num
                total_truth += len(truth)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_losses.append(loss_value / len(train_dataloader))
            all_accuracies.append(total_correct / total_truth)

            if val_dataloader is not None:
                total_correct_val = 0
                total_truth_val = 0
                loss_value_val = 0
                for batch in val_dataloader:
                    audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
                    with torch.no_grad():
                        audio_enc = encoder(audio_data)

                    pred = clf(audio_enc).view(-1, len(obj_list))  # (num_data, num_class)
                    truth = obj_id.view(-1).long()
                    loss = loss_func(pred, truth)
                    loss_value_val += loss.item()

                    truth = truth.detach().cpu().numpy()
                    pred_label = torch.argmax(pred, dim=-1).detach().cpu().numpy()
                    total_correct_val += np.sum(pred_label == truth)
                    total_truth_val += len(truth)

                all_losses_val.append(loss_value_val / len(val_dataloader))
                all_accuracies_val.append(total_correct_val / total_truth_val)

        return clf, {
            "all_losses_train": all_losses,
            "all_accuracies_train": all_accuracies,
            "all_losses_val": all_losses_val,
            "all_accuracies_val": all_accuracies_val
        }

    def test_linera_probe_clf(self, dataloader, encoder, clf, obj_list):
        total_correct_val = 0
        total_truth_val = 0
        all_truth = []
        all_pred = []
        for batch in dataloader:
            audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
            with torch.no_grad():
                audio_enc = encoder(audio_data)

            pred = clf(audio_enc).view(-1, len(obj_list))  # (num_data, num_class)
            truth = obj_id.view(-1).long()

            truth = truth.detach().cpu().numpy()
            pred_label = torch.argmax(pred, dim=-1).detach().cpu().numpy()
            total_correct_val += np.sum(pred_label == truth)
            total_truth_val += len(truth)
            all_truth.append(truth)
            all_pred.append(pred_label)

        accuracy = total_correct_val / total_truth_val

        return {
            "accuracy": accuracy,
            "all_truth": [int(p) for p in np.concatenate(all_truth).squeeze()],
            "all_pred": [int(p) for p in np.concatenate(all_pred).squeeze()]
        }
