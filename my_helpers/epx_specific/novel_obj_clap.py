import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from my_helpers.epx_specific.tune_clap import ClapPiplineAudio
from my_helpers.models import GeneralEncoder
from my_helpers.epx_specific.torch_pipeline import TransferPipeline


def clip_loss(sim_matrix):
    n = sim_matrix.shape[0]
    labels = torch.arange(n, device=sim_matrix.device)
    loss_i = nn.CrossEntropyLoss()(sim_matrix, labels)
    loss_t = nn.CrossEntropyLoss()(sim_matrix.T, labels)
    return (loss_i + loss_t) / 2


class ClapCrossObjPiplineAudio(ClapPiplineAudio):
    def __init__(self, context_dict: dict, audio_data_name: str, text_data_name: str):

        super().__init__(context_dict=context_dict,
                         novel_obj=True,
                         true_zero_shot=True,
                         audio_data_name=audio_data_name,
                         text_data_name=text_data_name)

        self.transfer_type = "CLAP_crossObj"

        self.encoder = None
        self.clf = None

    def learn_encoder(self, hyparams, one_batch, full_obj_list) -> dict:
        # ---------- make data loader ----------------
        shared_obj_list = self.context_dict['shared_object_list']
        shared_obj_label_map = {o: i for i, o in enumerate(shared_obj_list)}
        source_dataset = self.get_audio_text_dataset(
            tools=self.context_dict['source_tool_list'],
            behaviors=self.context_dict['source_beh_list'],
            objects=shared_obj_list,
            obj_label_map=shared_obj_label_map
        )
        target_dataset = self.get_audio_text_dataset(
            tools=self.context_dict['target_tool_list'],
            behaviors=self.context_dict['target_beh_list'],
            objects=shared_obj_list,
            obj_label_map=shared_obj_label_map
        )
        train_dataset = ConcatDataset([source_dataset, target_dataset])

        print(f"encoder train size: {len(train_dataset)}")

        batch_size = len(train_dataset) if one_batch else hyparams['batch_size']
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=hyparams['shuffle']
        )

        # ---------- setup model ----------------
        sample_data = self.audio_data_dict[
            self.context_dict['source_beh_list'][0]][self.context_dict['source_tool_list'][0]][
            self.context_dict['shared_object_list'][0]]
        sample_text = self.text_data_dict[
            self.context_dict['source_beh_list'][0]][self.context_dict['source_tool_list'][0]][
            self.context_dict['shared_object_list'][0]]
        data_dim = sample_data.shape[-1]
        text_dim = len(sample_text)

        # keep the project of the audio embedding the same dim as text
        audio_encoder = GeneralEncoder(input_size=data_dim, hidden_size=hyparams['encoder_hidden_dim'],
                                       output_size=text_dim, l2_norm=True).to(self.device)
        optimizer = torch.optim.AdamW(audio_encoder.parameters(), lr=hyparams['lr_encoder'])
        loss_func = clip_loss

        audio_encoder.train()

        # ---------- train ----------------
        all_losses = []
        for epoch in range(hyparams['epoch_encoder']):
            loss_value = 0
            for batch in train_dataloader:
                audio_data, text_data = batch['audio_data'].to(self.device), batch['text_data'].to(self.device)
                audio_enc = audio_encoder(audio_data)
                text_data = nn.functional.normalize(text_data, dim=-1, p=2)  # l2 norm

                sim_matrix = torch.matmul(audio_enc, text_data.T)  # (N,N)
                loss = loss_func(sim_matrix)
                # print(f"loss: {loss.item()}")
                loss_value += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_losses.append(loss_value / len(train_dataloader))

        self.encoder = audio_encoder

        return {
            "all_losses": all_losses
        }

    def test_retrival(self, test_obj_list: list, use_enc: bool, no_overlap=True) -> dict:
        if no_overlap:
            assert not bool(set(self.context_dict['shared_object_list']) & set(test_obj_list))

        obj_label_map = {o: i for i, o in enumerate(test_obj_list)}
        target_tool_list = self.context_dict['target_tool_list']
        target_beh_list = self.context_dict['target_beh_list']

        target_dataset = self.get_audio_text_dataset(
            tools=target_tool_list,
            behaviors=target_beh_list,
            objects=test_obj_list,
            obj_label_map=obj_label_map
        )
        print(f"clf test size: {len(target_dataset)}")

        test_dataloader = DataLoader(
            dataset=target_dataset,
            batch_size=len(target_dataset),
            shuffle=False
        )

        if use_enc:
            self.encoder.eval()

        total_correct = 0
        total_truth = 0
        all_truth, all_pred = [], []

        for batch in test_dataloader:
            audio_data = batch['audio_data'].to(self.device)
            obj_id, tool_id, beh_id = batch['obj_id'], batch['tool_id'], batch['beh_id']
            all_truth.extend(obj_id.tolist())

            objs = np.array([test_obj_list[i] for i in obj_id])
            tools = np.array([target_tool_list[i] for i in tool_id])
            behs = np.array([target_beh_list[i] for i in beh_id])

            if use_enc:
                with torch.no_grad():
                    audio_enc = self.encoder(audio_data)
                    audio_enc = nn.functional.normalize(audio_enc, dim=-1)
            else:
                audio_enc = nn.functional.normalize(audio_data, dim=-1)

            batch_size = len(audio_data)
            total_truth += batch_size

            for i in range(batch_size):
                gt_obj = objs[i]
                t, b = tools[i], behs[i]

                cand_emb = np.array([self.text_data_dict[b][t][o] for o in test_obj_list])
                cand_embs = torch.tensor(cand_emb, dtype=torch.float32).to(self.device)
                cand_embs = nn.functional.normalize(cand_embs, dim=-1)

                sims = torch.matmul(audio_enc[i], cand_embs.T)  # (num_test_obj,)
                pred_idx = sims.argmax().item()
                all_pred.append(pred_idx)

                pred_obj = test_obj_list[pred_idx]
                if pred_obj == gt_obj:
                    total_correct += 1

        accuracy = total_correct / total_truth

        return {
            "accuracy": accuracy,
            "all_truth": all_truth,
            "all_pred": all_pred
        }