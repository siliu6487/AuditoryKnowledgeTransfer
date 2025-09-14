import numpy as np
import torch
from torch.utils.data import DataLoader

from my_helpers.datasets import TrialSampler, ShuffledTrialSampler
from my_helpers.epx_specific.torch_pipeline import TransferPipeline
from my_helpers.models import LinearProbLayer, GeneralEncoder
from my_helpers.viz_helpers import viz_data
from sincere_loss_class import SINCERELoss


class Baseline1Audio(TransferPipeline):
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
        obj_list = self.context_dict['test_object_list']
        dataset = self.get_audio_dataset(
            tools=self.context_dict['target_tool_list'],
            behaviors=self.context_dict['target_beh_list'],
            objects=obj_list,
            obj_label_map={o: i for i, o in enumerate(obj_list)}
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

            sampler = ShuffledTrialSampler if hyparams['shuffle'] else TrialSampler
            batch_size = hyparams['batch_size'] if not one_batch else len(dataset)
            train_dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler(dataset, train_trials)
            )
            val_dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler(dataset, val_trials)
            )

            # ---------- train ----------------
            encoder, _ = self.train_enc(train_dataloader=train_dataloader, val_dataloader=None, hyparams=hyparams)
            clf, _ = self.train_clf(encoder=encoder, train_dataloader=train_dataloader,
                                    val_dataloader=None, hyparams=hyparams, obj_list=obj_list)
            test_result = self.test_clf(dataloader=val_dataloader, encoder=encoder, clf=clf, obj_list=obj_list)

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

    def train_clf(self, train_dataloader, val_dataloader, hyparams, obj_list, encoder):
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

    def test_clf(self, dataloader, encoder, clf, obj_list):
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
