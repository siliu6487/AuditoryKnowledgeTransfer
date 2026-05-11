import copy
import json
import os

import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, ConcatDataset
from itertools import chain

import configs
from my_helpers.datasets import TrialSampler, ShuffledTrialSampler
from my_helpers.epx_specific.shared_lalent_space import BasicSharedPiplineAudio
from my_helpers.epx_specific.torch_pipeline import TransferPipeline
from my_helpers.exp_helpers import general_exp_run_setup
from my_helpers.models import LinearProbLayer, GeneralEncoder
from my_helpers.viz_helpers import viz_data
from sincere_loss_class import SINCERELoss


class Baseline1Audio(TransferPipeline):
    def __init__(self, context_dict: dict, audio_data_name: str, all_tools: list = None, all_behs: list = None):
        """
        :param context_dict:
        :param audio_data_name:
        """
        super().__init__(transfer_type="B1_simple",
                         context_dict=context_dict,
                         modalities=['audio'],
                         audio_data_name=audio_data_name)
        self.encoder = None
        self.clf = None

        self.all_tools = all_tools if all_tools is not None else configs.ALL_TOOL_LIST
        self.all_behs = all_behs if all_behs is not None else configs.ALL_BEH_LIST
        self.context_combos = []

        self.context_dict = copy.deepcopy(context_dict)  # avoid changing the one outside

    def make_context_dict(self):
        combos = []
        self.context_combos = []
        for b in self.all_behs:
            for t in self.all_tools:
                combos.append([b, t])
        self.context_combos = combos

    def run_all_context_combos(self, k,
                               one_batch, shuffle, epoch_encoder, epoch_classifier,
                               num_test_fold, test_size, prev_test_fold, forbidden_test_obj_combo):
        self.make_context_dict()
        result_save_dir = os.path.join("test_result", self.transfer_type)
        print(f"➡️ result_save_dir: {result_save_dir}")

        full_results = {}
        for combo in self.context_combos:
            beh, tool = combo[0], combo[1]
            self.context_dict['target_beh_list'] = [beh]
            self.context_dict['target_tool_list'] = [tool]
            print(f"➡️ context COMBO: {combo}")

            setup = general_exp_run_setup(
                transfer_type=self.transfer_type, modality="audio", audio_data_name=self.audio_data_name,
                one_batch=one_batch, shuffle=shuffle, epoch_encoder=epoch_encoder,
                epoch_classifier=epoch_classifier, encoder_output_dim=None,
                num_test_fold=num_test_fold, test_size=test_size, prev_test_fold=prev_test_fold,
                forbidden_test_obj_combo=forbidden_test_obj_combo
            )
            hyparams = setup['hyparams']
            test_obj_lists = setup['test_obj_lists']
            results = setup['results']

            for i, test_objs in enumerate(test_obj_lists):

                self.context_dict['test_object_list'] = test_objs
                cv_result = self.cross_validate(k, hyparams, one_batch)

                accuracies = [cv_result[f'fold{fold + 1}']['accuracy'] for fold in range(k)]
                print(f"accuracy: {sum(accuracies) / len(accuracies) * 100:.2f}%. test objects :"
                      f"{self.context_dict['test_object_list']}")
                print(f"accuracy: {sum(accuracies) / len(accuracies):.4f} ")
                results['test_results']['accuracies'].append(sum(accuracies) / len(accuracies))

            a_l = results['test_results']['accuracies']
            avg_accuracy = sum(a_l) / len(a_l)
            print(f"avg accuracy: {avg_accuracy:.4f}")

            full_results[f"{beh}_{tool}"] = {
                "avg_accuracy": avg_accuracy,
                "accuracies": results['test_results']['accuracies']
            }

            # save intermediate results
            file_path = os.path.join(result_save_dir, "results1.json")
            with open(file_path, "w") as f:
                json.dump(full_results, f, indent=2)
            print(f"result saved to: {file_path}")

    def cross_validate(self, k: int, hyparams, one_batch):
        test_obj_list = self.context_dict['test_object_list']
        dataset = self.get_audio_dataset(
            tools=self.context_dict['target_tool_list'],
            behaviors=self.context_dict['target_beh_list'],
            objects=test_obj_list,
            obj_label_map={o: i for i, o in enumerate(test_obj_list)}
        )
        # cross validation
        num_trials = 10
        fold_size = num_trials // k
        all_trials = list(range(num_trials))
        cv_result = {f'fold{fold + 1}': {} for fold in range(0, k)}
        for fold in range(0, k):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < k - 1 else num_trials
            val_trials = list(range(start, end))
            train_trials = [t for t in all_trials if t not in val_trials]

            dataset_train = dataset.subset_by_trial_nums(train_trials)
            dataset_test = dataset.subset_by_trial_nums(val_trials)
            # print(f"train dataset: {len(dataset_train)}")

            train_dataloader = DataLoader(
                dataset_train,
                batch_size=hyparams['batch_size'] if not one_batch else len(dataset_train),
                shuffle=hyparams['shuffle']
            )

            test_dataloader = DataLoader(
                dataset_test,
                batch_size=hyparams['batch_size'] if not one_batch else len(dataset_test),
                shuffle=hyparams['shuffle']
            )

            # ---------- train ----------------
            encoder, _ = self.train_enc(train_dataloader=train_dataloader, val_dataloader=None, hyparams=hyparams)
            clf, _ = self.train_linear_probe_clf(encoder=encoder, train_dataloader=train_dataloader,
                                                 val_dataloader=None, hyparams=hyparams, obj_list=test_obj_list)
            test_result = self.test_linera_probe_clf(dataloader=test_dataloader, encoder=encoder,
                                                     clf=clf, obj_list=test_obj_list)

            cv_result[f'fold{fold + 1}']["accuracy"] = test_result['accuracy']
            cv_result[f'fold{fold + 1}']["all_truth"] = test_result['all_truth']
            cv_result[f'fold{fold + 1}']["all_pred"] = test_result['all_pred']

        return cv_result

    def train_enc(self, train_dataloader, val_dataloader, hyparams):
        # ---------- setup model ----------------
        sample_data = self.audio_data_dict[
            self.context_dict['source_beh_list'][0]][self.context_dict['source_tool_list'][0]][
            self.context_dict['shared_object_list'][0]]
        encoder = GeneralEncoder(input_size=sample_data.shape[-1], hidden_size=hyparams['encoder_hidden_dim'],
                                 output_size=hyparams['encoder_output_dim'], l2_norm=True).to(self.device)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=hyparams['lr_encoder'])
        loss_func = SINCERELoss(hyparams['temperature'])

        # ---------- train encoder----------------
        all_losses = []
        all_losses_val = []
        for epoch in range(hyparams['epoch_encoder']):
            loss_value = 0
            loss_value_val = 0
            for batch in train_dataloader:
                encoder.train()
                audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
                audio_enc = encoder(audio_data)

                loss = loss_func(audio_enc, obj_id)
                loss_value += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_losses.append(loss_value / len(train_dataloader))

            if val_dataloader is not None:
                for batch in val_dataloader:
                    encoder.eval()
                    audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
                    with torch.no_grad():
                        audio_enc = encoder(audio_data)

                    loss = loss_func(audio_enc, obj_id)
                    loss_value_val += loss.item()

                all_losses_val.append(loss_value_val / len(val_dataloader))

            return encoder, {
                "all_losses_train": all_losses,
                "all_losses_val": all_losses_val
            }

    def train_linear_probe_clf(self, train_dataloader, val_dataloader, hyparams, obj_list, encoder):
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
                # for batch in val_dataloader:
                #     audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
                #     with torch.no_grad():
                #         audio_enc = encoder(audio_data)
                #
                #     pred = clf(audio_enc).view(-1, len(obj_list))  # (num_data, num_class)
                #     truth = obj_id.view(-1).long()
                #     loss = loss_func(pred, truth)
                #     loss_value_val += loss.item()
                #
                #     truth = truth.detach().cpu().numpy()
                #     pred_label = torch.argmax(pred, dim=-1).detach().cpu().numpy()
                #     total_correct_val += np.sum(pred_label == truth)
                #     total_truth_val += len(truth)

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


class Baseline1WithSourceAudio(TransferPipeline):
    def __init__(self, context_dict: dict, audio_data_name: str):
        """
        :param context_dict:
        :param audio_data_name:
        """
        super().__init__(transfer_type="baseline1",
                         context_dict=context_dict,
                         modalities=['audio'],
                         audio_data_name=audio_data_name)
        self.encoder = None
        self.clf = None

    def cross_validate(self, k: int, hyparams, one_batch):
        full_obj_list = self.get_full_obj_list()
        test_obj_list = self.context_dict['test_object_list']
        # FIXME: proper baseline1

        enc_source_dataset = self.get_audio_dataset(
            tools=self.context_dict['source_tool_list'],
            behaviors=self.context_dict['source_beh_list'],
            objects=full_obj_list,
            obj_label_map={o: i for i, o in enumerate(full_obj_list)}
        )
        enc_target_dataset = self.get_audio_dataset(
            tools=self.context_dict['target_tool_list'],
            behaviors=self.context_dict['target_beh_list'],
            objects=full_obj_list,
            obj_label_map={o: i for i, o in enumerate(full_obj_list)}
        )

        # fair comparison to other multi-source tool scenes
        clf_source_dataset = self.get_audio_dataset(
            tools=self.context_dict['source_tool_list'],
            behaviors=self.context_dict['source_beh_list'],
            objects=test_obj_list,
            obj_label_map={o: i for i, o in enumerate(test_obj_list)}
        )

        clf_target_dataset = self.get_audio_dataset(
            tools=self.context_dict['target_tool_list'],
            behaviors=self.context_dict['target_beh_list'],
            objects=test_obj_list,
            obj_label_map={o: i for i, o in enumerate(test_obj_list)}
        )

        # cross validation
        num_trials = 10
        fold_size = num_trials // k
        all_trials = list(range(num_trials))
        cv_result = {
            f'fold{fold + 1}': {} for fold in range(0, k)
        }
        for fold in range(0, k):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < k - 1 else num_trials
            val_trials = list(range(start, end))
            train_trials = [t for t in all_trials if t not in val_trials]

            enc_target_dataset_train = enc_target_dataset.subset_by_trial_nums(train_trials)
            encoder_dataset = ConcatDataset([enc_source_dataset, enc_target_dataset_train])
            print(f"enc_target_dataset_train: {len(enc_target_dataset_train)}, encoder_dataset: {len(encoder_dataset)}")
            batch_size = hyparams['batch_size'] if not one_batch else len(encoder_dataset)
            enc_train_dataloader = DataLoader(
                encoder_dataset,
                batch_size=batch_size,
                shuffle=hyparams['shuffle']
            )

            clf_target_dataset_train = clf_target_dataset.subset_by_trial_nums(train_trials)
            clf_dataset_train = ConcatDataset([clf_source_dataset, clf_target_dataset_train])

            print(f"clf_target_dataset_train: {len(clf_target_dataset_train)}, clf_dataset_train: {len(clf_dataset_train)}")
            clf_train_dataloader = DataLoader(
                clf_dataset_train,
                batch_size=batch_size,
                shuffle=hyparams['shuffle']
            )
            clf_target_dataset_val = clf_target_dataset.subset_by_trial_nums(val_trials)
            print(f"clf_target_dataset_val: {len(clf_target_dataset_val)}")
            clf_val_dataloader = DataLoader(
                clf_target_dataset_val,
                batch_size=batch_size,
                shuffle=hyparams['shuffle']
            )

            # ---------- train ----------------
            encoder, _ = self.train_enc(train_dataloader=enc_train_dataloader, val_dataloader=None, hyparams=hyparams)
            clf, _ = self.train_linear_probe_clf(encoder=encoder, train_dataloader=clf_train_dataloader,
                                                 val_dataloader=None, hyparams=hyparams, obj_list=test_obj_list)
            test_result = self.test_linera_probe_clf(dataloader=clf_val_dataloader, encoder=encoder, clf=clf, obj_list=test_obj_list)

            cv_result[f'fold{fold + 1}']["accuracy"] = test_result['accuracy']
            cv_result[f'fold{fold + 1}']["all_truth"] = test_result['all_truth']
            cv_result[f'fold{fold + 1}']["all_pred"] = test_result['all_pred']

        return cv_result

    def train_enc(self, train_dataloader, val_dataloader, hyparams):
        # ---------- setup model ----------------
        sample_data = self.audio_data_dict[
            self.context_dict['source_beh_list'][0]][self.context_dict['source_tool_list'][0]][
            self.context_dict['shared_object_list'][0]]
        encoder = GeneralEncoder(input_size=sample_data.shape[-1], hidden_size=hyparams['encoder_hidden_dim'],
                                 output_size=hyparams['encoder_output_dim'], l2_norm=True).to(self.device)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=hyparams['lr_encoder'])
        loss_func = SINCERELoss(hyparams['temperature'])

        # ---------- train encoder----------------
        all_losses = []
        all_losses_val = []
        for epoch in range(hyparams['epoch_encoder']):
            loss_value = 0
            loss_value_val = 0
            for batch in train_dataloader:
                encoder.train()
                audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
                audio_enc = encoder(audio_data)

                loss = loss_func(audio_enc, obj_id)
                loss_value += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_losses.append(loss_value / len(train_dataloader))

            if val_dataloader is not None:
                for batch in val_dataloader:
                    encoder.eval()
                    audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
                    with torch.no_grad():
                        audio_enc = encoder(audio_data)

                    loss = loss_func(audio_enc, obj_id)
                    loss_value_val += loss.item()

                all_losses_val.append(loss_value_val / len(val_dataloader))

            return encoder, {
                "all_losses_train": all_losses,
                "all_losses_val": all_losses_val
            }

    def train_linear_probe_clf(self, train_dataloader, val_dataloader, hyparams, obj_list, encoder):
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
                # for batch in val_dataloader:
                #     audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
                #     with torch.no_grad():
                #         audio_enc = encoder(audio_data)
                #
                #     pred = clf(audio_enc).view(-1, len(obj_list))  # (num_data, num_class)
                #     truth = obj_id.view(-1).long()
                #     loss = loss_func(pred, truth)
                #     loss_value_val += loss.item()
                #
                #     truth = truth.detach().cpu().numpy()
                #     pred_label = torch.argmax(pred, dim=-1).detach().cpu().numpy()
                #     total_correct_val += np.sum(pred_label == truth)
                #     total_truth_val += len(truth)

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


class Baseline2Audio(BasicSharedPiplineAudio):
    """
    same as BasicSharedPiplineAudio but train encoder only on source data
    """
    def __init__(self, context_dict: dict, audio_data_name: str):
        """
        :param context_dict:
        :param audio_data_name:
        """
        super().__init__(context_dict=context_dict,
                         audio_data_name=audio_data_name)

        self.transfer_type = "baseline2"

    def train_encoder(self, hyparams, one_batch, full_obj_list) -> dict:
        # ---------- make data loader ----------------
        obj_label_map = {o: i for i, o in enumerate(full_obj_list)}
        dataset = self.get_audio_dataset(
            tools=self.context_dict['source_tool_list'],
            behaviors=self.context_dict['source_beh_list'],
            objects=full_obj_list,
            obj_label_map=obj_label_map
        )

        print(f"encoder train size: {len(dataset)}")

        batch_size = len(dataset) if one_batch else hyparams['batch_size']
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=hyparams['shuffle']
        )

        # ---------- setup model ----------------
        sample_data = self.audio_data_dict[
            self.context_dict['source_beh_list'][0]][self.context_dict['source_tool_list'][0]][
            self.context_dict['shared_object_list'][0]]
        encoder = GeneralEncoder(input_size=sample_data.shape[-1], hidden_size=hyparams['encoder_hidden_dim'],
                                 output_size=hyparams['encoder_output_dim'], l2_norm=True).to(self.device)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=hyparams['lr_encoder'])
        loss_func = SINCERELoss(hyparams['temperature'])

        encoder.train()

        # ---------- train ----------------
        all_losses = []
        for epoch in range(hyparams['epoch_encoder']):
            loss_value = 0
            for batch in train_dataloader:
                audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
                audio_enc = encoder(audio_data)

                loss = loss_func(audio_enc, obj_id)
                loss_value += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_losses.append(loss_value / len(train_dataloader))

        self.encoder = encoder

        return {
            "all_losses": all_losses
        }


