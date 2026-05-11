import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from my_helpers.models import GeneralEncoder, LinearProbLayer
from my_helpers.epx_specific.torch_pipeline import TransferPipeline
from sincere_loss_class import SINCERELoss


class BasicSharedPiplineAudio(TransferPipeline):
    def __init__(self, context_dict: dict, audio_data_name: str):

        super().__init__(transfer_type="shared",
                         context_dict=context_dict,
                         modalities=['audio'],
                         audio_data_name=audio_data_name)

        self.encoder = None
        self.clf = None

    def train_encoder(self, hyparams, one_batch, full_obj_list) -> dict:
        # ---------- make data loader ----------------
        obj_label_map = {o: i for i, o in enumerate(full_obj_list)}
        source_dataset = self.get_audio_dataset(
            tools=self.context_dict['source_tool_list'],
            behaviors=self.context_dict['source_beh_list'],
            objects=full_obj_list,
            obj_label_map=obj_label_map
        )
        target_dataset = self.get_audio_dataset(
            tools=self.context_dict['target_tool_list'],
            behaviors=self.context_dict['target_beh_list'],
            objects=self.context_dict['shared_object_list'],
            obj_label_map=obj_label_map
        )
        dataset = ConcatDataset([source_dataset, target_dataset])
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
                # print(f"loss: {loss.item()}")
                loss_value += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_losses.append(loss_value / len(train_dataloader))

        self.encoder = encoder

        return {
            "all_losses": all_losses
        }

    def train_classifier(self, hyparams, one_batch) -> dict:
        assert self.encoder is not None, "learned encoder not available."

        # ---------- make data loader ----------------
        obj_list = self.context_dict['test_object_list']
        obj_label_map = {o: i for i, o in enumerate(obj_list)}
        source_dataset = self.get_audio_dataset(
            tools=self.context_dict['source_tool_list'],
            behaviors=self.context_dict['source_beh_list'],
            objects=obj_list,
            obj_label_map=obj_label_map
        )

        print(f"clf train size: {len(source_dataset)}")
        batch_size = len(source_dataset) if one_batch else hyparams['batch_size']

        train_dataloader = DataLoader(
            dataset=source_dataset,
            batch_size=batch_size,
            shuffle=hyparams['shuffle']
        )

        # ---------- setup model ----------------
        clf = LinearProbLayer(in_dim=hyparams['encoder_output_dim'], num_classes=len(obj_list)).to(self.device)
        optimizer = torch.optim.AdamW(clf.parameters(), lr=hyparams['lr_classifier'])
        loss_func = torch.nn.CrossEntropyLoss()

        self.encoder.eval()

        # ---------- train ----------------
        all_losses = []
        all_accuracies = []
        for epoch in range(hyparams['epoch_classifier']):
            total_correct = 0
            total_truth = 0
            loss_value = 0
            for batch in train_dataloader:
                audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
                with torch.no_grad():
                    audio_enc = self.encoder(audio_data)

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
            all_accuracies.append(total_correct/total_truth)

        self.clf = clf
        return {
            "all_losses": all_losses,
            "all_accuracies": all_accuracies
        }

    def test_classifier(self) -> dict:
        assert self.encoder is not None and self.clf is not None, "need learned encoder and classifier"

        # ---------- make data loader ----------------
        test_obj_list = self.context_dict['test_object_list']
        obj_label_map = {o: i for i, o in enumerate(test_obj_list)}
        target_dataset = self.get_audio_dataset(
                    tools=self.context_dict['target_tool_list'],
                    behaviors=self.context_dict['target_beh_list'],
                    objects=test_obj_list,
                    obj_label_map=obj_label_map
        )
        print(f"clf test size: {len(target_dataset)}")
        test_dataloader = DataLoader(
            dataset=target_dataset,
            batch_size=len(target_dataset),
            shuffle=False
        )

        # ---------- prepare model ----------------
        self.encoder.eval()
        self.clf.eval()

        # ---------- test ----------------
        total_correct = 0
        total_truth = 0
        all_truth = []
        all_pred = []

        for batch in test_dataloader:
            audio_data, obj_id = batch['audio_data'].to(self.device), batch['obj_id'].to(self.device)
            with torch.no_grad():
                audio_enc = self.encoder(audio_data)
                pred = self.clf(audio_enc).view(-1, len(test_obj_list))

            truth = obj_id.view(-1).detach().cpu().numpy()
            pred_label = torch.argmax(pred, dim=-1).detach().cpu().numpy()
            correct_num = np.sum(pred_label == truth)

            total_correct += correct_num
            total_truth += len(truth)
            all_truth.append(truth)
            all_pred.append(pred_label)

        accuracy = total_correct/total_truth

        return {
            "accuracy": accuracy,
            "test_obj_list": test_obj_list,
            "all_truth": [int(p) for p in np.concatenate(all_truth).squeeze()],
            "all_pred": [int(p) for p in np.concatenate(all_pred).squeeze()]
        }




