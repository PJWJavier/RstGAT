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
        self.bertTokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # self.bartTokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    def beam_search(self, k, encoder,  max_length = 64):
        hidden_size =self.args.bert_dim
        vocab_size = self.args.vocab_size
        k_prev_words = torch.full((k, 1), 0, dtype=torch.long).to(self.args.device)  # (k, 1)
        # 此时输出序列中只有sos token
        seqs = k_prev_words  # (k, 1)
        # 初始化scores向量为0
        top_k_scores = torch.zeros(k, 1)
        complete_seqs = list()
        complete_seqs_scores = list()
        step = 1
        encoder = encoder.repeat(k,1,1)
        # hidden = torch.zeros(1, k, hidden_size)  # h_0: (1, k, hidden_size)
        while True:
            outputs = self.model(encoder_hidden=encoder, decoder_input_ids=k_prev_words)
            outputs = outputs.logits# outputs: (k, seq_len, vocab_size)
            next_token_logits = outputs[:, -1, :]  # (k, vocab_size)
            if step == 1:
                # 因为最开始解码的时候只有一个结点<sos>,所以只需要取其中一个结点计算topk
                top_k_scores, top_k_words = next_token_logits[0].topk(k, dim=0, largest=True, sorted=True)
            else:
                # 此时要先展开再计算topk，如上图所示。
                # top_k_scores: (k) top_k_words: (k)
                top_k_scores, top_k_words = next_token_logits.contiguous().view(-1).topk(k, 0, True, True)
            prev_word_inds = top_k_words / vocab_size
            prev_word_inds = prev_word_inds.long()# (k)  实际是beam_id
            next_word_inds = top_k_words % vocab_size  # (k)  实际是token_id
            # seqs: (k, step) ==> (k, step+1)
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            # 当前输出的单词不是eos的有哪些(输出其在next_wod_inds中的位置, 实际是beam_id)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != 2]
            # 输出已经遇到eos的句子的beam id(即seqs中的句子索引)
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())  # 加入句子
                complete_seqs_scores.extend(top_k_scores[complete_inds])  # 加入句子对应的累加log_prob
            # 减掉已经完成的句子的数量，更新k, 下次就不用执行那么多topk了，因为若干句子已经被解码出来了
            k -= len(complete_inds)

            if k == 0:  # 完成
                break

            # 更新下一次迭代数据, 仅专注于那些还没完成的句子
            seqs = seqs[incomplete_inds]
            # hidden = hidden[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)  # (s, 1) s < k
            k_prev_words = torch.cat([k_prev_words[incomplete_inds], next_word_inds[incomplete_inds].unsqueeze(1)], dim=1) # (s, 1) s < k
            encoder = encoder[incomplete_inds]
            if step > max_length:  # decode太长后，直接break掉
                break
            step += 1
        i = complete_seqs_scores.index(max(complete_seqs_scores))  # 寻找score最大的序列
        # 有些许问题，在训练初期一直碰不到eos时，此时complete_seqs为空
        seq = torch.tensor(complete_seqs[i])
        seq = self.args.tokenizer.decode(seq, skip_special_tokens=True)
        return seq

    def generate_summary(self,test_samples, model):
        tokenizer = self.args.tokenizer
        inputs = tokenizer(
            test_samples,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        outputs = model.generate(input_ids, attention_mask=attention_mask)
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs, output_str

    def train(self, Trainset: DataLoader, Evalset: DataLoader):
        max_test_precision, max_f1, max_recall= 0, 0, 0
        global_step = 0
        for epoch in range(self.args.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0

            with tqdm(total=len(Trainset.dataset.data)) as pbar:
                for i_batch, sample_batched in enumerate(Trainset):
                    # print('train batch num:{}'.format(i_batch))
                    global_step += 1
                    self.model.train()
                    self.optimizer.zero_grad()
                    plain_article = sample_batched["plain_article"]
                    rst = sample_batched["rst"]
                    rst = [r.to(self.args.device) for r in rst]
                    input_tensor = sample_batched["input_tensor"].to(self.args.device)
                    sent_num = sample_batched["sent_num"]
                    reference_tensor = sample_batched["reference_tensor"].to(self.args.device)
                    ref_segment_tensor = sample_batched["ref_segment_tensor"].to(self.args.device)
                    segment_tensor = sample_batched["segment_tensor"].to(self.args.device)
                    tokenizer = self.args.tokenizer

                    # mymodel = BartForConditionalGeneration.from_pretrained('facebook/bart-base', cache_dir='./BART',
                    #                                                        local_files_only=True).to(self.args.device)
                    # output = mymodel(input_tensor[0],segment_tensor[0])
                    # encoder_hidden = output.encoder_last_hidden_state[0]
                    # generated = self.gen(encoder_hidden)
                    # outputs = output.logits[0]
                    # input = self.args.tokenizer.decode(input_tensor[0][0])

                    # if i_batch == 3:
                    #     print(1)
                    try:
                        for i in range(2):
                            inputs = tokenizer(
                                plain_article[i],
                                padding="max_length",
                                truncation=True,
                                max_length=1024,
                                return_tensors="pt",
                            )
                            input_ids = inputs.input_ids.to(self.args.device)
                            attention_mask = inputs.attention_mask.to(self.args.device)
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                            model_output = outputs.logits
                    except:
                        print(input_tensor)
                        continue
                    # output = model_output[0]
                    # encoder_hidden = model_output[1]
                    # generated = self.gen(encoder_hidden[0])
                    # outputs = model_output.logits[0]
                    # input = self.args.tokenizer.decode(input_tensor[0][0])
                    # text = self.gen(input_tensor)
                    # train_ground_truth = sample_batched[self.args.inputs_cols[4]].to(self.args.device)
                    # loss_backward
                    model_output = model_output[0]
                    ground_truth = []
                    mask = reference_tensor != 1
                    input_ids = torch.masked_select(reference_tensor, mask).view(-1, 1)
                    reference_tensor = torch.roll(reference_tensor, -1, dims=1)
                    for i in range(len(model_output)):
                        ground_truth.append(reference_tensor[i])

                    model_pred = torch.cat(model_output, dim=0).reshape(-1, self.args.vocab_size)
                    ground_truth = torch.cat(ground_truth, dim=0).reshape(1, -1).squeeze()
                    mask = ground_truth != 1
                    ground_truth = torch.masked_select(ground_truth, mask)

                    mask = mask.repeat(self.args.vocab_size, 1).T
                    model_pred = torch.masked_select(model_pred, mask).view(-1,self.args.vocab_size)



                    pred_text = torch.argmax(model_pred,dim=1)
                    # ground_truth = self.bertTokenizer.decode(ground_truth)
                    loss = self.criterion(model_pred, ground_truth.long())
                    # assert model_output.shape[0] == polarity.shape[0]
                    # loss = self.criterion(model_output, polarity.long())
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    n_correct += (torch.argmax(model_pred, -1) == ground_truth).sum().item()
                    n_total += ground_truth.shape[0]
                    pbar.update(Trainset.batch_size)

            senti_train_precision = n_correct / n_total
            print('avg_loss: {:.4f}'.format(loss.item()))
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
                    model_path = 'last_state_dict/f1_{0}'.format(round(eval_f1, 4))
                    torch.save(self.model.state_dict(), model_path)
                    print('>> saved: ' + model_path)
            else:
                print('eval_f1: {:.4f}, eval_precision: {:.4f}, eval_recall: {:.4f}'.format(eval_f1, eval_precision, eval_recall))
            print('-' * 100)
        model_path = 'last_state_dict/f1_{0}'.format(round(eval_f1, 4))
        torch.save(self.model.state_dict(), model_path)
        print('>> saved: ' + model_path)

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
        other_case = pd.DataFrame(columns=["input","output","ground_truth"])
        with torch.no_grad():
            with tqdm(total=len(Evalset.dataset.data)) as pbar:
                for t_batch, t_sample_batched in enumerate(Evalset):
                    # if t_batch > 3:
                    #     break
                    # switch model to training mode, clear gradient accumulators
                    rst = t_sample_batched["rst"]
                    rst = [r.to(self.args.device) for r in rst]
                    input_tensor = t_sample_batched["input_tensor"].to(self.args.device)
                    sent_num = t_sample_batched["sent_num"]
                    reference_tensor = t_sample_batched["reference_tensor"].to(self.args.device)
                    ref_segment_tensor = t_sample_batched["ref_segment_tensor"].to(self.args.device)
                    segment_tensor = t_sample_batched["segment_tensor"].to(self.args.device)
                    # print(t_batch)
                    # if t_batch == 33:
                    #     print(1)
                    try:
                        model_output = self.model(input_tensor, segment_tensor,reference_tensor,ref_segment_tensor, sent_num, rst)
                    except:
                        print(t_batch)
                        continue
                    encoder_hidden = model_output[1]
                    outputs = []
                    try:
                        for encoder_out in  encoder_hidden:
                            generated = self.beam_search(k=5,encoder = encoder_out)
                            # generated = self.gen(encoder_out)
                            outputs.append(generated)
                    except:
                        continue

                    # ground_truth = polarity
                    ground_truth = []
                    for i in range(len(model_output)):
                        ground_truth = reference_tensor[i]
                        model_pred = outputs[i]
                        # if model_pred.shape[0] != ground_truth.shape[0]:
                        #     print(t_batch)
                        #     continue
                        # assert model_output.shape[0] == polarity.shape[0]
                        # n_senti_correct += (torch.argmax(model_pred, -1) == ground_truth).sum().item()
                        mask = ground_truth != 1
                        ground_truth = torch.masked_select(ground_truth, mask)
                        Bart_input_tensor = input_tensor[i].view(1,-1)
                        mask = Bart_input_tensor != 1
                        Bart_input_tensor = torch.masked_select(Bart_input_tensor, mask)
                        model_pred = model_pred.replace('<s>',"").replace('</s>',"")
                        input = self.args.tokenizer.decode(Bart_input_tensor).replace('<s>',"").replace('</s>',"")
                        ground_truth = self.args.tokenizer.decode(ground_truth).replace('<s>',"").replace('</s>',"")
                        res = rouge.get_scores(model_pred, ground_truth)
                        sari = SARI.SARIsent(input, model_pred, ground_truth)
                        all_sari.append(sari)
                        eval_f1, eval_test_precision, eval_test_recall = res[0]['rouge-1']['f'], res[0]['rouge-1']['p'], res[0]['rouge-1']['r']
                        if eval_test_precision > 0.6:
                            good_case.loc[len(good_case)] = [input, model_pred, ground_truth]
                        elif eval_test_precision < 0.3:
                            bad_case.loc[len(bad_case)] = [input, model_pred, ground_truth]
                        else:
                            other_case.loc[len(other_case)] = [input, model_pred, ground_truth]
                        all_f1.append(eval_f1)
                        all_pre.append(eval_test_precision)
                        all_recall.append(eval_test_recall)
                        all_test_correct += n_senti_correct
                    pbar.update(Evalset.batch_size)
                # all_test_total += polarity
        # eval_result = torch.argmax(model_pred_all, -1).cpu()
        # ground_truth_all = ground_truth_all.view(-1).cpu()
        # n_correct = (eval_result == ground_truth_all).sum().item()
        good_case.to_excel("samples/good_case.xlsx")
        bad_case.to_excel("samples/bad_case.xlsx")
        other_case.to_excel("samples/other_case.xlsx")
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