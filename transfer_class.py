# %%
import copy
import logging
import os
import pickle
import sys
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

import configs
import model
import my_helpers.data_helpers
from my_helpers.data_helpers import sanity_check_data_labels, train_test_split_by_trials, fill_missing_hyper_params
from my_helpers.viz_helpers import plot_learning_progression, plot_data_histogram
from sincere_loss_class import SINCERELoss


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# %%
class ToolKnowledgeTransfer:
    def __init__(self, encoder_loss_fuc, data_name, multi_class):
        """
        :param encoder_loss_fuc: "TL" for triplet loss or "sincere"
        :param: data_name: "audio_16kHz_token_down32_beh3.bin" for behavior3 only, tokenized audio by BEATS model,
                            downsized by 16 and flattened to 1D.
                           "dataset_discretized.bin" for discretized data, flattened to 1D
        :param: multi_class: classifier predicts object (multi-class) or object attributes (multi-label)
        """
        ####load dataset
        assert encoder_loss_fuc in ['TL', 'sincere', 'mulsupcon']

        robots_data_filepath = SCRIPT_DIR + os.sep + 'data' + os.sep + data_name
        bin_file = open(robots_data_filepath, 'rb')
        self.orig_data = pickle.load(bin_file)
        bin_file.close()

        if configs.use_tool_emb:
            self._get_tool_emb_dict(configs.tool_emb_name)
            # self._add_tool_emb(configs.tool_emb_name)

        sanity_check_data_labels(self.orig_data)
        self.encoder_loss_fuc = encoder_loss_fuc
        self.enc_l2_norm = self._decide_l2_norm()
        logging.info(f"Encoder loss function: {encoder_loss_fuc}")
        logging.info(f"Encoder l2 norm: {self.enc_l2_norm}")

        #### load names
        data_file_path = os.sep.join([SCRIPT_DIR, 'data', 'dataset_metadata.bin'])
        bin_file = open(data_file_path, 'rb')
        metadata = pickle.load(bin_file)
        bin_file.close()

        self.behaviors = list(metadata.keys())
        self.objects = metadata[self.behaviors[0]]['objects']
        self.tools = metadata[self.behaviors[0]]['tools']
        self.trials = metadata[self.behaviors[0]]['trials']
        logging.debug(f"behaviors:, {len(self.behaviors)}, {self.behaviors}")
        logging.debug(f"objects: , {len(self.objects)}, {self.objects}")
        logging.debug(f"tools: , {len(self.tools)}, {self.tools}")
        logging.debug(f'trials: , {len(self.trials)}, {self.trials}')

        ####
        self.input_dim = 0
        self.num_clf_class = 0
        self.multi_class = multi_class
        if multi_class:
            self.clf_loss = torch.nn.CrossEntropyLoss()  # for multi-class classification, softmax applied
        else:
            self.clf_loss = torch.nn.BCEWithLogitsLoss()  # for multi-label classification, sigmoid applied
            csv_path = r'data' + os.sep + "Object_Attribute_Table.csv"
            self.obj_attribute_table = pd.read_csv(csv_path)
            self.attribute_dict = self.obj_attribute_table.set_index('Object').to_dict(orient='index')
            self.obj_attributes = self.obj_attribute_table.columns[1:].tolist()
            self.num_attributes = len(self.obj_attributes)
            self.num_clf_class = self.num_attributes
        logging.info(f"{'multi-class' if multi_class else 'multi-label'} classifier; "
                     f"loss function: {type(self.clf_loss).__name__}")

        ###
        self.trained_encoder = None
        self.trained_clf = None
        self.encoder_output_dim = None

    def _decide_l2_norm(self):
        """might change this rule later"""
        return self.encoder_loss_fuc == "sincere"

    def _assign_labels_to_data(self, structured_data: torch.Tensor, object_list: list) -> torch.Tensor:
        """
        structured_data shape: [n_behavior, n_tools, len(object_list), n_trials, data_sample_dim]
        return label shape: [n_behavior, n_tools, len(object_list), n_trials]"""
        truth = np.zeros_like(structured_data[:, :, :, :, -1])
        for i in range(len(object_list)):
            truth[:, :, i, :] = i  # all trials (fourth dimension) have the same object label
        return truth

    def _prepare_data_classifier(self, Encoder, behavior_list, source_tool_list, new_object_list,
                                 modality_list, trail_list, trial_val_portion, use_tool_emb) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return:
        data shape: [n_behavior, n_tools, num_objects, n_trials, emb_dim],
        label shape: [n_behavior*n_tools*num_objects&n_trials, ]
        """
        logging.debug(f"➡️ prepare_data_classifier..")

        logging.debug(f"get source data for classifier from {source_tool_list}: {new_object_list}")
        source_data, truth = self.get_data(behavior_list, source_tool_list, modality_list, new_object_list, trail_list,
                                           get_labels=True, use_tool_emb=use_tool_emb).values()
        if self.multi_class:  # restart index of labels from 0
            truth = self._assign_labels_to_data(truth, new_object_list)

        if Encoder is not None:
            with torch.no_grad():
                source_data = torch.tensor(source_data, dtype=configs.data_dtype, device=configs.device)
                if self.multi_class:
                    Encoder.l2_norm = self.enc_l2_norm
                    encoded_source = Encoder(source_data)
                else:
                    encoded_source = Encoder.encode(source_data.reshape(-1, self.input_dim))
                encoded_source = encoded_source.detach().cpu().numpy()
        else:
            encoded_source = source_data

        split_dict = train_test_split_by_trials(source_data=encoded_source, truth_source=truth,
                                                trial_list=trail_list, trial_val_portion=trial_val_portion)

        logging.debug(f"train_truth: \n      {split_dict['truth_source_train']}")

        if self.multi_class:
            split_dict['truth_source_train'] = split_dict['truth_source_train'].reshape(-1)
            if trial_val_portion > 0:
                split_dict['truth_source_val'] = split_dict['truth_source_val'].reshape(-1)
        else:
            split_dict['truth_source_train'] = split_dict['truth_source_train'].reshape(-1, self.num_clf_class)
            if trial_val_portion > 0:
                split_dict['truth_source_val'] = split_dict['truth_source_val'].reshape(-1, self.num_clf_class)
            logging.debug(f"val_truth: \n      {split_dict['truth_source_val']}")

        return (split_dict['source_data_train'], split_dict['source_data_val'],
                split_dict['truth_source_train'], split_dict['truth_source_val'])

    def train_classifier(self, Encoder, trail_list=configs.trail_list, modality_list=configs.modality_list,
                         new_object_list=configs.new_object_list, source_beh_list=configs.source_beh_list,
                         source_tool_list=configs.source_tool_list, use_tool_emb=configs.use_tool_emb,
                         hyparams=None, plot_learning=True, save_fig=True):
        if self.multi_class:
            self.num_clf_class = len(new_object_list)
        my_helpers.data_helpers.set_torch_seed()
        logging.debug(f"➡️ train_classifier..")
        hyparams = fill_missing_hyper_params(hyparams=hyparams, param_model="classifier")

        train_encoded_source, val_encoded_source, train_truth_flat, val_truth_flat = self._prepare_data_classifier(
            Encoder=Encoder, behavior_list=source_beh_list, source_tool_list=source_tool_list,
            new_object_list=new_object_list, modality_list=modality_list, trail_list=trail_list,
            trial_val_portion=hyparams['trial_val_portion'], use_tool_emb=use_tool_emb)
        Classifier = model.classifier(input_size=train_encoded_source.shape[-1],
                                      output_size=self.num_clf_class).to(configs.device)
        optimizer = optim.AdamW(Classifier.parameters(), lr=hyparams['lr_classifier'])
        loss_record = np.zeros([2, hyparams['epoch_classifier']])
        best_loss_val = np.inf
        prev_loss_val = np.inf
        prev_loss_tr = np.inf
        best_clf = copy.deepcopy(Classifier)
        best_epoch = 0
        patience_counter = 0
        cross_val = hyparams['trial_val_portion'] > 0
        for epoch in range(hyparams['epoch_classifier']):
            pred_tr = Classifier(train_encoded_source)
            pred_flat_tr = pred_tr.view(-1, self.num_clf_class)  # (num_data, num_class)
            loss_tr = self.clf_loss(pred_flat_tr, train_truth_flat)
            loss_record[0, epoch] = loss_tr.detach().cpu().numpy()

            if cross_val:
                with torch.no_grad():
                    pred_val = Classifier(val_encoded_source)
                    pred_flat_val = pred_val.view(-1, self.num_clf_class)
                    loss_val = self.clf_loss(pred_flat_val, val_truth_flat)
                    loss_record[1, epoch] = loss_val.detach().cpu().numpy()

            optimizer.zero_grad()
            loss_tr.backward()
            optimizer.step()

            if (epoch + 1) % 500 == 0:
                pred_label = torch.argmax(pred_flat_tr, dim=-1)
                correct_num = torch.sum(pred_label == train_truth_flat)
                accuracy_train = correct_num / len(train_truth_flat)

                if cross_val:
                    pred_label = torch.argmax(pred_flat_val, dim=-1)
                    correct_num = torch.sum(pred_label == val_truth_flat)
                    accuracy_val = correct_num / len(val_truth_flat)
                    logging.info(
                        f"epoch {epoch + 1}/{hyparams['epoch_classifier']}, train loss: {loss_tr.item():.4f}, "
                        f"train accuracy: {accuracy_train.item() * 100 :.2f}%, "
                        f"val loss: {loss_val.item():.4f}, val accuracy: {accuracy_val.item() * 100 :.2f}%, "
                        f"random guess accuracy: {100 / len(new_object_list):.2f}%")
                else:
                    logging.info(f"epoch {epoch + 1}/{hyparams['epoch_classifier']}, train loss: {loss_tr.item():.4f}, "
                                 f"train accuracy: {accuracy_train.item() * 100 :.2f}%, val accuracy: None,"
                                 f"random guess accuracy: {100 / len(new_object_list):.2f}%")

            if cross_val:  # cross validation
                if loss_val < best_loss_val:  # save best model with best val loss
                    best_loss_val = loss_val
                    best_clf = copy.deepcopy(Classifier)  # deep copy the model at this time
                    best_epoch = epoch
                else:
                    if epoch + 1 == hyparams['epoch_classifier']:  # last epoch, take the best model
                        Classifier = best_clf
                if hyparams['early_stop_patience_clf'] is not None:  # compare previous loss with current loss
                    patience_counter = 0 if loss_val + hyparams[
                        'clf_tolerance'] <= prev_loss_val else patience_counter + 1
                    prev_loss_val = loss_val  # for the next epoch
                    if patience_counter >= hyparams['early_stop_patience_clf']:
                        logging.debug(f"Early stopping triggered at epoch {epoch + 1}. "
                                      f"Best validation loss: {best_loss_val}, best epoch: {best_epoch + 1}")
                        Classifier = best_clf  # restore the best encoder
                        loss_record = loss_record[:, :epoch + 1]
                        break  # stop training
            else:  # stop by training loss
                if hyparams['early_stop_patience_clf'] is not None:  # compare previous loss with current loss
                    patience_counter = 0 if loss_tr + hyparams[
                        'clf_tolerance'] <= prev_loss_tr else patience_counter + 1
                    prev_loss_tr = loss_tr  # for the next epoch
                    if patience_counter >= hyparams['early_stop_patience_clf']:
                        logging.debug(f"Early stopping triggered at epoch {epoch + 1}.")
                        loss_record = loss_record[:, :epoch + 1]
                        break  # stop training

        if plot_learning:
            plot_learning_progression(record=loss_record, type='classifier', lr_classifier=hyparams['lr_classifier'],
                                      encoder_output_dim=hyparams['encoder_output_dim'],
                                      save_fig=save_fig, loss_func=self.encoder_loss_fuc,
                                      encoder_hidden_dim=None, TL_margin=None, sincere_temp=None,
                                      lr_encoder=None, save_name=f'classifier_{self.encoder_loss_fuc}')
        self.trained_clf = Classifier

        return Classifier

    def eval_classifier(self, Encoder, Classifier, behavior_list=configs.target_beh_list,
                        tool_list=configs.target_tool_list, new_object_list=configs.new_object_list,
                        modality_list=configs.modality_list, trail_list=configs.trail_list,
                        use_tool_emb=configs.use_tool_emb, return_pred=False) -> dict:
        logging.debug(f"➡️ eval..")
        logging.debug(f"{tool_list}: {new_object_list}")
        target_data, truth = self.get_data(behavior_list, tool_list, modality_list, new_object_list, trail_list,
                                           get_labels=True, use_tool_emb=use_tool_emb).values()

        if self.multi_class:
            truth_flat = self._assign_labels_to_data(truth, object_list=new_object_list).flatten()
        else:
            truth_flat = truth.reshape(-1, self.num_clf_class).astype(int)
        # for t in range(len(tool_list)):
        #     for o in range(len(new_object_list)):
        #         start = t * (len(new_object_list) * len(trail_list)) + o * len(trail_list)
        #         truth_flat[start: start + len(trail_list)] = o
        target_data = torch.tensor(target_data, dtype=configs.data_dtype, device=configs.device)
        with torch.no_grad():
            if Encoder is not None:
                if self.multi_class:
                    Encoder.l2_norm = self.enc_l2_norm
                    encoded_target = Encoder(target_data)
                else:
                    encoded_target = Encoder.encode(target_data.reshape(-1, self.input_dim))
            else:
                encoded_target = target_data
            pred = Classifier(encoded_target)
        pred_flat = pred.view(-1, self.num_clf_class)
        if self.multi_class:
            per_label_accuracy = None
            pred_label = torch.argmax(pred_flat, dim=-1).detach().cpu().numpy()
            correct_num = np.sum(pred_label == truth_flat)
            accuracy_test = correct_num / len(truth_flat)
            logging.info(
                f"test accuracy: {accuracy_test * 100:.2f}%, random guess accuracy: {100 / len(new_object_list):.2f}%")
        else:
            pred_flat = torch.sigmoid(pred_flat).detach().cpu().numpy()
            pred_label = (pred_flat > 0.5).astype(int)
            accuracy_test = np.all(pred_label == truth_flat, axis=1).mean()  # all match accuracy
            per_label_accuracy = {}
            for label_idx, label in enumerate(self.obj_attribute_table.columns[1:]):
                correct = (pred_label[:, label_idx] == truth_flat[:, label_idx]).sum()
                total = truth_flat.shape[0]
                per_label_accuracy[label] = correct / total

        truth_flat = truth_flat.tolist()
        pred_label = pred_label.tolist()
        return {"test_accuracy": accuracy_test, "per_label_accuracy": per_label_accuracy,
                "pred": pred_label, "label": truth_flat}

    def train_encoder(self, source_tool_list=configs.source_tool_list, target_tool_list=configs.target_tool_list,
                      source_beh_list=configs.source_beh_list, target_beh_list=configs.target_beh_list,
                      old_object_list=configs.old_object_list, new_object_list=configs.new_object_list,
                      modality_list=configs.modality_list, trail_list=configs.enc_trial_list,
                      use_tool_emb=configs.use_tool_emb, hyparams=None, plot_learning=True):
        """
        :param new_object_list: list of objects that only source tool has
        :param old_object_list: list of objects that both tools share
        :param source_tool_list: tool(s) with data from old_object_list + new_object_list
        :param target_tool_list: tool(s) with data from  old_object_list
        :param source_beh_list:
        :param target_beh_list:
        :param modality_list:
        :param trail_list: the index of training trails, e.g. [0,1,2,3,4,5,6,7]
        :param plot_learning: whether plot learning progression
        :param hyparams: a dictionary of hyperparameters, missing parameters will be filled with values from configs
        :return: trained encoder, either from the last epoch, or the best one based on validation set
        """
        hyparams = fill_missing_hyper_params(hyparams=hyparams, param_model="encoder")

        logging.debug(f"➡️ train_encoder..")
        loss_record = np.zeros([2, hyparams['epoch_encoder']])  # save train and val losses over epochs

        source_data, target_data, truth_source, truth_target = self._get_encoder_data_and_labels(
            source_beh_list=source_beh_list, target_beh_list=target_beh_list,
            trail_list=trail_list, modality_list=modality_list,
            source_tool_list=source_tool_list, target_tool_list=target_tool_list,
            shared_object_list=old_object_list, novel_object_list=new_object_list, use_tool_emb=use_tool_emb)
        n_class = truth_source.shape[-1]
        if n_class == source_data.shape[-2]:
            n_class = 1
        print(f"source_data: {source_data.shape}")
        print(f"truth_source: {truth_source.shape}, n_class: {n_class}")

        splits = train_test_split_by_trials(source_data=source_data, target_data=target_data,
                                            truth_source=truth_source, truth_target=truth_target,
                                            trial_list=trail_list, trial_val_portion=hyparams['trial_val_portion'])
        '''
        If we have more than one modality, we may need preprocessing and the input dim may not the 
        sum of data dim across all considered modalities. But I just put it here because we have 
        not figured out what to do.
        '''
        self.encoder_output_dim = hyparams['encoder_output_dim']
        self.input_dim = 0
        self.input_dim = len(splits['source_data_train'][0, 0, 0, 0])

        my_helpers.data_helpers.set_torch_seed()
        Encoder = model.encoder(input_size=self.input_dim, hidden_size=hyparams['encoder_hidden_dim'],
                                    output_size=hyparams['encoder_output_dim'],
                                    l2_norm=self.enc_l2_norm).to(configs.device)
        optimizer = optim.AdamW(Encoder.parameters(), lr=hyparams['lr_encoder'])
        # TODO: why is sincere's val loss lower than train?
        best_loss_val = np.inf
        best_enc = copy.deepcopy(Encoder)
        best_epoch = 0
        patience_counter = 0
        wind_idx = 1
        cross_val = hyparams['trial_val_portion'] > 0
        print(f"trial cross validation for encoder training: {cross_val}")
        for epoch in range(hyparams['epoch_encoder']):
            if self.encoder_loss_fuc == "TL":
                loss = self._TL_loss_fn(splits['source_data_train'], splits['target_data_train'],
                                        Encoder, alpha=hyparams['TL_margin'],
                                        encoder_output_dim=hyparams['encoder_output_dim'],
                                        pairs_per_batch_per_object=hyparams['pairs_per_batch_per_object'])
                if cross_val:
                    with torch.no_grad():
                        loss_val = self._TL_loss_fn(splits['source_data_val'], splits['target_data_val'],
                                                    Encoder, alpha=hyparams['TL_margin'],
                                                    encoder_output_dim=hyparams['encoder_output_dim'],
                                                    pairs_per_batch_per_object=hyparams['pairs_per_batch_per_object'])
            elif self.encoder_loss_fuc == "sincere":
                loss = self._sincere_loss_fn(splits['source_data_train'], splits['truth_source_train'],
                                             splits['target_data_train'], splits['truth_target_train'],
                                             Encoder, temperature=hyparams['sincere_temp'],
                                             encoder_output_dim=hyparams['encoder_output_dim'])

                if cross_val:
                    with torch.no_grad():
                        loss_val = self._sincere_loss_fn(splits['source_data_val'], splits['truth_source_val'],
                                                         splits['target_data_val'], splits['truth_target_val'],
                                                         Encoder, temperature=hyparams['sincere_temp'],
                                                         encoder_output_dim=hyparams['encoder_output_dim'])

            else:
                raise Exception(f"{self.encoder_loss_fuc} not available.")
            loss_record[0, epoch] = loss.detach().cpu().numpy()
            if cross_val:
                loss_record[1, epoch] = loss_val.detach().cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 500 == 0:
                logging.info(f"epoch {epoch + 1}/{hyparams['epoch_encoder']}, loss: {loss.item():.4f}")

            if cross_val:  # cross validation
                if loss_val < best_loss_val:  # save best model with best val loss
                    best_loss_val = loss_val
                    best_enc = copy.deepcopy(Encoder)  # deep copy the model at this time
                    best_epoch = epoch
                else:
                    if epoch + 1 == hyparams['epoch_encoder']:  # last epoch, take the best model
                        Encoder = best_enc
                # early stopping by smoothed windows, not epochs
                if hyparams['early_stop_patience_enc'] is not None:
                    window_size = hyparams['smooth_wind_size']
                    if epoch + 1 >= window_size * 2 and (epoch + 1) % window_size == 0:
                        wind_idx += 1  # compare the last two windows
                        wind_prev = np.mean(loss_record[1, window_size * (wind_idx - 2):window_size * (wind_idx - 1)])
                        wind_curr = np.mean(loss_record[1, window_size * (wind_idx - 1):window_size * wind_idx])
                        patience_counter = 0 if wind_curr + hyparams['tolerance'] <= wind_prev else patience_counter + 1
                        logging.debug(f"epoch: {epoch + 1}, patience_counter: {patience_counter}")
                    if patience_counter >= hyparams['early_stop_patience_enc']:
                        logging.debug(f"Early stopping triggered at epoch {epoch + 1}. "
                                      f"Best validation loss: {best_loss_val}, best epoch: {best_epoch + 1}")
                        Encoder = best_enc  # restore the best encoder
                        loss_record = loss_record[:, :epoch + 1]
                        break  # stop training

        if plot_learning:
            plot_learning_progression(record=loss_record, type='encoder', loss_func=self.encoder_loss_fuc,
                                      lr_classifier=None, encoder_output_dim=hyparams['encoder_output_dim'],
                                      encoder_hidden_dim=hyparams['encoder_hidden_dim'],
                                      TL_margin=hyparams['TL_margin'], sincere_temp=hyparams['sincere_temp'],
                                      lr_encoder=hyparams['lr_encoder'], save_name=f'encoder_{self.encoder_loss_fuc}')
        self.trained_encoder = Encoder
        return Encoder

    def _sincere_loss_fn(self, source_data, truth_source, target_data, truth_target,
                         Encoder, temperature, encoder_output_dim) -> torch.Tensor:
        all_embeds_norm, all_labels = self._make_encoder_projections(
            Encoder=Encoder, source_data=source_data, target_data=target_data, truth_source=truth_source,
            truth_target=truth_target, encoder_output_dim=encoder_output_dim)
        # FIXME: don't initialize loss everytime!
        sincere_loss = SINCERELoss(temperature)
        if len(all_labels.shape) > 1:
            all_labels = torch.squeeze(all_labels)  # labels (torch.tensor): (B,)
        return sincere_loss(all_embeds_norm, all_labels)

    def _get_same_object_list(self, encoded_source, encoded_target, encoder_output_dim):
        """
        input data shape: [n_behavior, n_tools, num_objects, n_trials, data_sample_dim]
        assuming that encoded_source is constructed in the order of old_object_list + new_object_list,
        and encoded_target in the order of old_object_list, so their old object part match
        """
        same_object_list = []
        tot_len = encoded_source.shape[2]
        target_len = encoded_target.shape[2]
        for i in range(tot_len):
            if i < target_len:
                object_list1 = encoded_source[:, :, i, :, :].reshape([-1, encoder_output_dim])
                object_list2 = encoded_target[:, :, i, :, :].reshape([-1, encoder_output_dim])
                object_list = torch.concat([object_list1, object_list2], dim=0)
            else:
                object_list = encoded_source[:, :, i, :, :].reshape([-1, encoder_output_dim])
            same_object_list.append(object_list)

        return same_object_list

    def _TL_loss_fn(self, source_data, target_data, Encoder, alpha, encoder_output_dim,
                    pairs_per_batch_per_object) -> torch.Tensor:
        Encoder.l2_norm = self.enc_l2_norm
        encoded_source = Encoder(source_data)
        encoded_target = Encoder(target_data)  # TODO: fix problem for empty target
        same_object_list = self._get_same_object_list(encoded_source, encoded_target, encoder_output_dim)

        trail_tot_num_list = np.array([same_object_list[i].shape[0] for i in range(len(same_object_list))])
        tot_object_num = encoded_source.shape[2]

        A_mat = torch.zeros(pairs_per_batch_per_object * tot_object_num, encoder_output_dim, device=configs.device)
        P_mat = torch.zeros(pairs_per_batch_per_object * tot_object_num, encoder_output_dim, device=configs.device)
        N_mat = torch.zeros(pairs_per_batch_per_object * tot_object_num, encoder_output_dim, device=configs.device)

        for object_index in range(tot_object_num):
            object_list = same_object_list[object_index]

            # Sample anchor and positive
            A_index = np.random.choice(trail_tot_num_list[object_index], size=pairs_per_batch_per_object)
            P_index = np.random.choice(trail_tot_num_list[object_index], size=pairs_per_batch_per_object)
            start = object_index * pairs_per_batch_per_object
            end = (object_index + 1) * pairs_per_batch_per_object
            A_mat[start: end] = object_list[A_index, :]
            P_mat[start: end] = object_list[P_index, :]

            # Sample negative
            N_object_list = np.random.choice(len(trail_tot_num_list), size=pairs_per_batch_per_object)
            N_list = torch.zeros(pairs_per_batch_per_object, encoder_output_dim,
                                 dtype=configs.data_dtype).to(configs.device)
            for i in range(len(N_object_list)):
                N_object_index = N_object_list[i]
                N_trail_index = np.random.choice(trail_tot_num_list[N_object_index])
                N_list[i] = same_object_list[N_object_index][N_trail_index]
                N_mat[start: end] = N_list

        dPA = torch.norm(A_mat - P_mat, dim=1)
        dNA = torch.norm(A_mat - N_mat, dim=1)

        d = dPA - dNA + alpha
        d[d < 0] = 0

        loss = torch.mean(d)
        return loss

    def get_data(self, behavior_list, tool_list, modality_list, object_list, trail_list,
                 get_labels=False, use_tool_emb=False) -> dict:
        """
        :return:
        data shape: None, or tensor of shape [n_behavior, n_tools, num_objects, n_trials, data_sample_dim],
        label shape: if get_labels, None or tensor of shape
                        [n_behavior, n_tools, num_objects, n_trials, 1 or num_attributes]

        for each behavior&tool, data is ordered by object_list, then trail_list
        we ASSUME the dataset labels are created using sorted 15 object names, i.e., "cane-suger" is 0.
        Note that the returned labels do NOT always start from 0
            i.e., object_list is a subset of the objects list used to create the labels in the original dataset
        """
        logging.debug(f"➡️get_data...")

        meta_data = {b: {t: {} for t in tool_list} for b in behavior_list}

        # ASSUMPTION: only one modality
        if len(modality_list) == 1 and behavior_list and tool_list and object_list and trail_list:
            data_dim = len(self.orig_data[behavior_list[0]][tool_list[0]][modality_list[0]][object_list[0]]['X'][0])
            if use_tool_emb:
                toll_emb_dim = len(self.tool_emb_dict[tool_list[0]].shape)
                if toll_emb_dim == 2:
                    data_dim += len(self.tool_emb_dict[tool_list[0]][0])
                elif toll_emb_dim == 1:
                    data_dim += len(self.tool_emb_dict[tool_list[0]])
                else:
                    raise Exception(f"tool emb shape too high dimensional: {toll_emb_dim}")
            data = np.zeros((len(behavior_list), len(tool_list), len(object_list), len(trail_list), data_dim))
            label_len = 1 if self.multi_class else self.num_attributes
            label = np.zeros((len(behavior_list), len(tool_list), len(object_list), len(trail_list), label_len))
            '''
            Now we have 1 behavior, 1 tool. The data dim is 1x1xtrail_num x data_dim
            But this can work for multiple behaviors, tools
            '''
            for behavior_index, behavior in enumerate(behavior_list):
                for tool_index, tool in enumerate(tool_list):
                    # if use_tool_emb:
                        # print(f"tool emb shap: {self.tool_emb_dict[tool].shape}")
                    for object_index, object_name in enumerate(object_list):
                        meta_data[behavior][tool][object_name] = len(trail_list)
                        for trial_index in range(len(trail_list)):
                            try:
                                trial = trail_list[trial_index]
                                trial_data = self.orig_data[behavior][tool][modality_list[0]][object_name]['X'][trial]
                                if use_tool_emb:
                                    if use_tool_emb == "random_tool_trial":  # cat random tool trial number
                                        rand_trial = np.random.randint(0, 10)
                                        trial_data = np.concatenate((trial_data, self.tool_emb_dict[tool][rand_trial]))
                                    elif use_tool_emb is True:
                                        if len(self.tool_emb_dict[tool].shape) == 2:  # cat same tool trial number
                                            trial_data = np.concatenate((trial_data, self.tool_emb_dict[tool][trial]))
                                        else:
                                            trial_data = np.concatenate((trial_data, self.tool_emb_dict[tool]))
                                    else:
                                        raise Exception(f"use_tool_emb not recognized: {use_tool_emb}")
                                    # print(f"trial_data shape: {trial_data.shape}")
                                data[behavior_index][tool_index][object_index][trial_index] = trial_data
                                if get_labels:
                                    if self.multi_class:  # multi-class labels, integer class labels
                                        label[behavior_index][tool_index][object_index][trial_index] = \
                                            self.orig_data[behavior][tool][modality_list[0]][object_name]['Y'][trial]
                                    else:  # multi-label labels, vector labels
                                        object_attributes = self.attribute_dict.get(object_name, {})
                                        label[behavior_index][tool_index][object_index][trial_index] = \
                                            np.array(list(object_attributes.values()))

                            except Exception as e:
                                print(f"something wrong here: behavior: {behavior}, tool: {tool}, "
                                      f"modality: {modality_list[0]}, object: {object_name}, trail index: {trial_index}")
                                raise e

            data = data.astype(np.dtype(str(configs.data_dtype).split('.')[1]))
            # FIXME: scale tool embedding and sensormotor embeddings
            # plot_data_histogram(data.flatten())
            label = label.astype(np.dtype(str(configs.label_dtype).split('.')[1]))

            logging.debug(f"data mata: {meta_data}")
            logging.debug(f"structured data shape: {data.shape} from tool(s): {tool_list}")
            logging.debug(f"structured label shape: {label.shape}")

        else:
            data = None
            label = None
            logging.debug("No data.")
            '''
            if we have more than one modality, the data dim are different and a tensor cannot hold this.
            So I leave this for future extension.
            '''
        return {"data": data, "label": label}

    def _get_tool_emb_dict(self, tool_emb_name):
        tool_emb_filepath = r'data' + os.sep + tool_emb_name
        if ".bin" in tool_emb_name:
            bin_file = open(tool_emb_filepath, 'rb')
            self.tool_emb_dict = pickle.load(bin_file)
            bin_file.close()
        elif ".npz" in tool_emb_name:
            loaded = np.load(tool_emb_filepath)
            self.tool_emb_dict = {k: loaded[k] for k in loaded.files}
        else:
            raise Exception(f"Not able to read this file: {tool_emb_name}")

    # def _add_tool_emb(self, tool_emb_name):
    #     tool_emb_filepath = r'data' + os.sep + tool_emb_name
    #     bin_file = open(tool_emb_filepath, 'rb')
    #     self.tool_emb_dict = pickle.load(bin_file)
    #     for tool in self.tool_emb_dict.keys():
    #         self.orig_data[behavior][tool][modality_list[0]][object_name]['X'][trial]
    #     for beh, beh_dict in self.orig_data.items():
    #         tool_dict = beh_dict[tool]
    #         for mode, mode_dict in tool_dict.items():
    #             for obj, obj_dict in mode_dict.items():
    #                 obj_dict['X'] =
    def _get_encoder_data_and_labels(self, source_beh_list, target_beh_list,
                                     source_tool_list, target_tool_list, modality_list,
                                     shared_object_list, novel_object_list, trail_list, use_tool_emb):
        """
        shared_object_list: typically old_object_list, but it can be any overlapping objects between source and target
        novel_object_list: typically new_object_list, but it can be any  non-overlapping objects
        :return:
        data shape: [n_behavior, n_tools, num_objects, n_trials, data_sample_dim],
        label shape: [n_behavior, n_tools, num_objects, n_trials, 1], always starts from 0

         source_data: data from source tool & shared_object_list + novel_object_list
         truth_source: labels for source tool & shared_object_list + novel_object_list
                        index starts from 0 to len(shared_object_list + novel_object_list) - 1
         target_data: data from shared_object_list + novel_object_list
         truth_target: labels for target tool & shared_object_list, index starts from 0 to len(shared_object_list) - 1
        """
        logging.debug(f"➡️ get_encoder_data_and_labels...")
        assert (source_beh_list or target_beh_list) and trail_list and modality_list
        # assert len(behavior_list) == 1  # for now, one behavior only
        all_obj_list = shared_object_list + novel_object_list  # has to be this order
        # =========  get source tool data
        logging.debug(f"get source data for encoder: {source_tool_list}: {all_obj_list}")
        source_data, truth_source = self.get_data(source_beh_list, source_tool_list, modality_list, all_obj_list,
                                                  trail_list, use_tool_emb=use_tool_emb, get_labels=True).values()

        # ========= get target tool data, get shared object by default
        logging.debug(f"get target data for encoder: {target_tool_list}: {shared_object_list}")
        target_data, truth_target = self.get_data(target_beh_list, target_tool_list, modality_list, shared_object_list,
                                                  trail_list, use_tool_emb=use_tool_emb, get_labels=True).values()

        # assign multi class labels from 0 to num_objects-1
        if source_data is not None:
            if self.multi_class:
                truth_source = self._assign_labels_to_data(structured_data=source_data, object_list=all_obj_list)
            else:  # multi_label
                raise Exception(f"make sure that we don't need to reset labels")
        if target_data is not None:
            if self.multi_class:
                truth_target = self._assign_labels_to_data(structured_data=target_data, object_list=shared_object_list)
            else:  # multi_label
                raise Exception(f"make sure that we don't need to reset labels")

        return source_data, target_data, truth_source, truth_target

    def _make_encoder_projections(self, Encoder, source_data, target_data, truth_source, truth_target,
                                  encoder_output_dim) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        encode structured data [n_behavior, n_tools, num_objects, n_trials, data_sample_dim] ,
        and flatten the embeddings to [num_samples, emb_dim] , i.e., [n_behavior*n_tools*num_objects*n_trials, emb_dim]
        :return all_embeds: [num_samples, emb_dim], all_labels: [num_samples, 1]
        """
        Encoder.l2_norm = self.enc_l2_norm
        if source_data is not None:
            encoded_source = Encoder(source_data).reshape(-1, encoder_output_dim)
            truth_source = truth_source.reshape(-1, 1)
        else:
            encoded_source = torch.empty((0, encoder_output_dim)).to(configs.device)
            truth_source = torch.empty((0, 1), dtype=configs.label_dtype).to(configs.device)

        if target_data is not None:
            encoded_target = Encoder(target_data).reshape(-1, encoder_output_dim)
            truth_target = truth_target.reshape(-1, 1)
        else:
            encoded_target = torch.empty((0, encoder_output_dim)).to(configs.device)
            truth_target = torch.empty((0, 1), dtype=configs.label_dtype).to(configs.device)
        # concat source and target
        all_labels = torch.cat([truth_source, truth_target], dim=0)
        all_embeds = torch.cat([encoded_source, encoded_target], dim=0)

        return all_embeds, all_labels

    # %%
