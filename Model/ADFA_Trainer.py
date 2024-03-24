import pandas as pd
import torch
from  tqdm import tqdm
import numpy as np
import time
import os
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BartTokenizer
import torch.nn as nn
import torch.nn.functional as F
import Utils.SARI as SARI
from transformers import BertTokenizer, BartForConditionalGeneration
from rouge import Rouge




class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, scheduler, device):
        super(Trainer, self).__init__()
        self.args       = args
        # Training Component
        self.model       = model
        self.criterion   = criterion
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        # Training Params
        self.log_step    = args.log_step
        self.epoch       = 0
        self.step        = 0
        self.best        = -float('inf')
        self.train_loss  = []
        self.eval_loss   = []
        self.all_acc     = []
        self.all_f1      = []
        self.eval_inform = {'loss' : [], 'acc' : [], 'precision' : [], 'recall' : [], 'f1_score' : []}
        self.bert_path = 'dataset/BERT/'
        # self.bertTokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # self.bartTokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    def gen(self, encoder, decoder_input_ids=None):
        tokenizer = self.args.tokenizer
        device = self.device
        list_v = []
        list_ids = []
        while True:
            # 使用 tokenizer 把输入文本和已经生成的文本转换为模型可以接受的输入格式
            # input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
            encoder_hidden = encoder[0]
            if decoder_input_ids == None:
                decoder_input_ids = torch.tensor([102, 101]).unsqueeze(0).to(device)
                list_ids.append(torch.tensor([102, 102]))
                list_ids.append(torch.tensor([101, 101]))
                list_v.append(torch.tensor([1., 1.]))
                list_v.append(torch.tensor([1., 1.]))

            # 输入到模型中进行生成
            outputs = self.model(encoder_hidden=encoder_hidden, decoder_input_ids=decoder_input_ids)
            # 获取模型的输出
            logits = outputs.logits
            # 获取模型输出的下一个单词的 ID
            next_word_id = logits[0, -1, :].argmax(-1).unsqueeze(0).unsqueeze(0)
            if next_word_id[0][0].item() == 102:
                decoder_input_ids = torch.cat([decoder_input_ids, next_word_id], dim=-1)
                break

            tmp = torch.nn.functional.softmax(logits[0, -1, :], dim=-1)
            k = 2
            topk_v, topk_ids = tmp.topk(k, dim=-1)
            list_v.append(topk_v)
            list_ids.append(topk_ids)
            # for i in range(k):
            #     print(topk_v[i].item(),tokenizer.decode(topk_ids[i]))

            # 将这个 ID 添加到已经生成的文本的 ID 中
            decoder_input_ids = torch.cat([decoder_input_ids, next_word_id], dim=-1)
            if decoder_input_ids.shape[1] >=128:
                break
        # 使用 tokenizer 解码模型的输出
        generated_prefix = tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)
        return list_v, list_ids, generated_prefix

    def train(self, Trainset: DataLoader, Evalset: DataLoader):
        max_test_precision, max_f1, max_recall= 0, 0, 0
        global_step = 0
        for epoch in range(self.args.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(Trainset):
                # print('train batch num:{}'.format(i_batch))
                global_step += 1
                self.model.train()
                self.optimizer.zero_grad()
                rst = sample_batched["rst"]
                polarity = sample_batched["polarity"]
                map = sample_batched["map"]
                rst = [r.to(self.args.device) for r in rst]
                map = [torch.tensor(m).to(self.args.device) for m in map]
                input_tensor = sample_batched["input_tensor"].to(self.args.device)
                sent_num = sample_batched["sent_num"]
                ref_num = sample_batched["ref_num"]
                reference_tensor = sample_batched["reference_tensor"].to(self.args.device)
                ref_segment_tensor = sample_batched["ref_segment_tensor"].to(self.args.device)
                segment_tensor = sample_batched["segment_tensor"].to(self.args.device)
                # if i_batch == 3:
                #     print(1)
                model_output = self.model(input_tensor, segment_tensor,reference_tensor,ref_segment_tensor, sent_num, ref_num, rst, polarity, map)
                model_output = model_output[0]
                # text = self.gen(input_tensor)
                # train_ground_truth = sample_batched[self.args.inputs_cols[4]].to(self.args.device)
                # loss_backward
                ground_truth = []
                for i in range(len(model_output)):
                    ground_truth.append(reference_tensor[i][:ref_num[i]])
                error = False
                for i in range(len(model_output)):
                    if model_output[i].shape[0] !=  ground_truth[i].shape[0]:
                        error = True
                if error:
                    print(i_batch)
                    continue
                model_pred = torch.cat(model_output, dim=0).reshape(-1, self.args.vocab_size)
                ground_truth = torch.cat(ground_truth, dim=0).reshape(1, -1).squeeze()

                mask = ground_truth != 0
                ground_truth = torch.masked_select(ground_truth, mask)
                mask = mask.repeat(self.args.vocab_size, 1).T
                model_pred = torch.masked_select(model_pred, mask).view(-1,self.args.vocab_size)

                # model_pred = self.bertTokenizer.decode(model_pred)
                # ground_truth = self.bertTokenizer.decode(ground_truth)
                loss = self.criterion(model_pred, ground_truth.long())
                # assert model_output.shape[0] == polarity.shape[0]
                # loss = self.criterion(model_output, polarity.long())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                if global_step % self.args.log_step == 0:
                    n_correct += (torch.argmax(model_pred, -1) == ground_truth).sum().item()
                    n_total += ground_truth.shape[0]
                    senti_train_precision = n_correct / n_total
                    print('loss: {:.4f}'.format(loss.item()))
                    print('train_precision: {:.4f}'.format(senti_train_precision))
                    # switch model to evaluation mode
                    self.model.eval()
                    eval_precision, eval_f1, eval_recall = self.eval(Evalset)
                    if eval_f1 > max_f1:
                        max_test_precision = eval_precision
                        max_f1 = eval_f1
                        max_recall = eval_recall
                        # max_result = eval_result.numpy()
                        # self.save_eval(Evalset, max_result)
                        print('>>>f1:{:.4f},precision:{:.4f},recall:{:.4f}'.
                              format(eval_f1, eval_precision, eval_recall))
                        print('>>>max_f1:{:.4f},max_precision:{:.4f},max_recall:{:.4f}'.
                              format(max_f1, max_test_precision, eval_recall))
                        if self.args.save_dic:
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            model_path = 'state_dict/f1{0}'.format(round(eval_f1, 4))
                            torch.save(self.model.state_dict(), model_path)
                            print('>> saved: ' + model_path)
                    else:
                        print('eval_f1: {:.4f}, eval_precision: {:.4f}, eval_recall: {:.4f}'.format(eval_f1, eval_precision, eval_recall))
                    print('-' * 100)
        return max_test_precision, max_f1, max_recall

    def eval(self, Evalset: DataLoader):
        all_test_correct, all_test_total = 0, 0
        ground_truth_all, model_pred_all = None, None
        n_senti_correct = 0
        self.model.eval()
        rouge = Rouge()
        all_f1, all_pre, all_recall = [], [], []
        all_sari = []
        good_case = pd.DataFrame(columns=["input","output","ground_truth"])
        bad_case = pd.DataFrame(columns=["input","output","ground_truth"])
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(Evalset):
                # switch model to training mode, clear gradient accumulators
                rst = t_sample_batched["rst"]
                polarity = t_sample_batched["polarity"]
                map = t_sample_batched["map"]
                rst = [r.to(self.args.device) for r in rst]
                map = [torch.tensor(m).to(self.args.device) for m in map]
                input_tensor = t_sample_batched["input_tensor"].to(self.args.device)
                sent_num = t_sample_batched["sent_num"]
                ref_num = t_sample_batched["ref_num"]
                reference_tensor = t_sample_batched["reference_tensor"].to(self.args.device)
                ref_segment_tensor = t_sample_batched["ref_segment_tensor"].to(self.args.device)
                segment_tensor = t_sample_batched["segment_tensor"].to(self.args.device)
                # print(t_batch)
                # if t_batch == 33:
                #     print(1)

                model_output = self.model(input_tensor, segment_tensor,reference_tensor,ref_segment_tensor, sent_num, ref_num, rst, polarity, map)
                encoder_hidden = model_output[1]
                generated = self.gen(encoder_hidden)
                model_pred = model_output
                # ground_truth = polarity
                ground_truth = []
                for i in range(len(model_output)):
                    ground_truth.append(reference_tensor[i][:ref_num[i]])
                model_pred = generated[2][0]
                ground_truth = torch.cat(ground_truth, dim=0).reshape(1, -1).squeeze()
                # if model_pred.shape[0] != ground_truth.shape[0]:
                #     print(t_batch)
                #     continue
                # assert model_output.shape[0] == polarity.shape[0]
                # n_senti_correct += (torch.argmax(model_pred, -1) == ground_truth).sum().item()
                mask = ground_truth != 0
                ground_truth = torch.masked_select(ground_truth, mask)
                input_tensor = input_tensor.view(1,-1)
                mask = input_tensor != 0
                input_tensor = torch.masked_select(input_tensor, mask)
                input = self.args.tokenizer.decode(input_tensor).replace('[SEP]',"").replace('[CLS]',"")
                ground_truth = self.args.tokenizer.decode(ground_truth).replace('[SEP]',"").replace('[CLS]',"")
                res = rouge.get_scores(model_pred, ground_truth)
                sari = SARI.SARIsent(input, model_pred, ground_truth)
                all_sari.append(sari)
                eval_f1, eval_test_precision, eval_test_recall = res[0]['rouge-1']['f'], res[0]['rouge-1']['p'], res[0]['rouge-1']['r']
                if eval_test_precision > 0.9:
                    good_case.loc[len(good_case)] = [input, model_pred, ground_truth]
                elif eval_test_precision < 0.5:
                    bad_case.loc[len(bad_case)] = [input, model_pred, ground_truth]
                all_f1.append(eval_f1)
                all_pre.append(eval_test_precision)
                all_recall.append(eval_test_recall)
                all_test_correct += n_senti_correct
                # all_test_total += polarity
        # eval_result = torch.argmax(model_pred_all, -1).cpu()
        # ground_truth_all = ground_truth_all.view(-1).cpu()
        # n_correct = (eval_result == ground_truth_all).sum().item()
        good_case.to_excel("samples/good_case.xlsx")
        bad_case.to_excel("samples/bad_case.xlsx")
        eval_f1, eval_test_precision, eval_test_recall =sum(all_f1)/len(all_f1) , sum(all_pre)/len(all_pre) , sum(all_recall)/len(all_recall)
        eval_sari = sum(all_sari)/len(all_sari)
        print('eval_sari: {:.4f}'.format(eval_sari))
        # if self.args.polarity_dim == 2:
        #     eval_f1 = metrics.f1_score(ground_truth_all, eval_result)
        #     eval_test_precision = metrics.precision_score(ground_truth_all,eval_result)
        #     eval_test_recall = metrics.recall_score(ground_truth_all, eval_result)
        # else:
        #     eval_f1 = metrics.f1_score(ground_truth_all, eval_result, average='macro')
        #     eval_test_precision = metrics.precision_score(ground_truth_all, eval_result, average='macro')
        #     eval_test_recall = metrics.recall_score(ground_truth_all, eval_result, average='macro')
        return eval_test_precision, eval_f1, eval_test_recall

    def save_eval(self,Evalset,result):
        Eval_dataset = Evalset.dataset.data
        output = []
        all_text = []
        for i in range(len(Eval_dataset)):
            output.append(Eval_dataset[i]["all_text"])
            all_text.append(Eval_dataset[i]["all_text"])
            output.append(Eval_dataset[i]["polarity"])
            output.append(result[i])
            output.append(1 if Eval_dataset[i]["polarity"] == result[i] else 0)
            output.append(Eval_dataset[i]["cxg1"])
            output.append(Eval_dataset[i]["cxg2"])
        columns = ["sentences", "ground_truth", "pred", "correct", "cxg1", "cxg1"]
        output = np.array(output)
        output = output.reshape(len(Eval_dataset), 6)
        output = pd.DataFrame(output, columns=columns)
        output.to_excel("Evaluate/Results/"+ self.args.task_name + "_Results.xlsx", index=0)
        return