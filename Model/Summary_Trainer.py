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
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast
import torch.distributed as dist
from transformers.modeling_outputs import BaseModelOutput
class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty # 长度惩罚的指数系数
        self.num_beams = num_beams # beam size
        self.beams = [] # 存储最优序列及其累加的log_prob score
        self.worst_score = 1e9 # 将worst_score初始为无穷大。

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty # 计算惩罚后的score
        if len(self) < self.num_beams or score > self.worst_score:
                # 如果类没装满num_beams个序列
                # 或者装满以后，但是待加入序列的score值大于类中的最小值
                # 则将该序列更新进类中，并淘汰之前类中最差的序列
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                                # 如果没满的话，仅更新worst_score
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
                # 当解码到某一层后, 该层每个结点的分数表示从根节点到这里的log_prob之和
                # 此时取最高的log_prob, 如果此时候选序列的最高分都比类中最低分还要低的话
                # 那就没必要继续解码下去了。此时完成对该句子的解码，类中有num_beams个最优序列。
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, rank):
        super(Trainer, self).__init__()
        self.args       = args
        # Training Component
        self.model       = model
        self.criterion   = criterion
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        self.rank        = rank
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

    def train(self, Trainset: DataLoader, Evalset: DataLoader):
        max_test_precision, max_f1, max_recall= 0, 0, 0
        global_step = 0
        scaler = GradScaler()

        for epoch in range(self.args.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0

            with tqdm(total=len(Trainset.dataset.data) // self.args.gpus) as pbar:
                for i_batch, sample_batched in enumerate(Trainset):
                    with autocast():
                    # print('train batch num:{}'.format(i_batch))
                        global_step += 1
                        self.model.train()
                        self.optimizer.zero_grad()
                        rst = sample_batched["rst"].to(self.args.device)
                        input_tensor = sample_batched["input_tensor"].to(self.args.device)
                        sent_num = sample_batched["sent_num"]
                        reference_tensor = sample_batched["reference_tensor"].to(self.args.device)
                        ref_segment_tensor = sample_batched["ref_segment_tensor"].to(self.args.device)
                        segment_tensor = sample_batched["segment_tensor"].to(self.args.device)

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
                            model_output = self.model(input_tensor, segment_tensor,reference_tensor,ref_segment_tensor, sent_num, rst,mode="train")
                        except Exception as e:
                            print(i_batch)
                            print(e)
                            continue
                        # output = model_output[0]
                        # encoder_hidden = model_output[1]
                        # generated = self.gen(encoder_hidden[0])
                        # outputs = model_output.logits[0]
                        # input = self.args.tokenizer.decode(input_tensor[0][0])
                        # text = self.gen(input_tensor)
                        # train_ground_truth = sample_batched[self.args.inputs_cols[4]].to(self.args.device)
                        # loss_backward
                        ground_truth = []
                        mask = reference_tensor != 1
                        input_ids = torch.masked_select(reference_tensor, mask).view(-1, 1)
                        reference_tensor = torch.roll(reference_tensor, -1, dims=1)
                        for i in range(len(reference_tensor)):
                            ground_truth.append(reference_tensor[i])

                        model_pred = torch.cat(model_output, dim=0).reshape(-1, self.args.vocab_size-1)
                        ground_truth = torch.cat(ground_truth, dim=0).reshape(1, -1).squeeze()
                        mask = ground_truth != 1
                        ground_truth = torch.masked_select(ground_truth, mask)
                        mask = mask.repeat(self.args.vocab_size-1, 1).T
                        model_pred = torch.masked_select(model_pred, mask).view(-1,self.args.vocab_size-1)

                        # pred_text = torch.argmax(model_pred,dim=1)
                        # ground_truth = self.bertTokenizer.decode(ground_truth)
                        loss = self.criterion(model_pred, ground_truth.long())
                        # assert model_output.shape[0] == polarity.shape[0]
                        # loss = self.criterion(model_output, polarity.long())

                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    # loss.backward()
                    # self.optimizer.step()
                        self.scheduler.step()
                    n_correct += (torch.argmax(model_pred, -1) == ground_truth).sum().item()
                    n_total += ground_truth.shape[0]
                    pbar.update(Trainset.batch_size)

            train_precision = n_correct / n_total
            print('loss: {:.4f}'.format(loss.item()))
            print('train_precision: {:.4f}'.format(train_precision))
            # switch model to evaluation mode
            self.model.eval()
            if train_precision>0.6:
                if self.rank == 0:
                    if self.args.save_dic:
                        if not os.path.exists('state_dict'):
                            os.mkdir('state_dict')
                        model_path = 'state_dict/pre_{0}_checkpoint.pth'.format(round(train_precision, 4))
                        checkpoint = {
                            "model_state_dict": self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            "epoch": epoch,
                            "scheduler":self.scheduler.state_dict()
                        }
                        torch.save(checkpoint, model_path)
                        print('>> saved: ' + model_path)
                    print('-' * 100)

                eval_precision, eval_f1, eval_recall = self.eval(Evalset)

                eval_precision = torch.tensor(eval_precision,dtype=torch.float, device=self.args.device)
                eval_precision_list = torch.zeros(self.args.world_size, dtype=torch.float, device=self.args.device)
                dist.all_gather_into_tensor(eval_precision_list, eval_precision)
                eval_precision_list = eval_precision_list.to("cpu").numpy().tolist()
                eval_precision = sum(eval_precision_list) / len(eval_precision_list)

                eval_f1 = torch.tensor(eval_f1, dtype=torch.float, device=self.args.device)
                eval_f1_list = torch.zeros(self.args.world_size, dtype=torch.float, device=self.args.device)
                dist.all_gather_into_tensor(eval_f1_list, eval_f1)
                eval_f1_list = eval_f1_list.to("cpu").numpy().tolist()
                eval_f1 = sum(eval_precision_list) / len(eval_f1_list)

                eval_recall = torch.tensor(eval_recall, dtype=torch.float, device=self.args.device)
                eval_recall_list = torch.zeros(self.args.world_size, dtype=torch.float, device=self.args.device)
                dist.all_gather_into_tensor(eval_recall_list, eval_recall)
                eval_recall_list = eval_recall_list.to("cpu").numpy().tolist()
                eval_recall = sum(eval_precision_list) / len(eval_recall_list)

                if self.rank == 0:
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
                            model_path = 'state_dict/f1_{0}_checkpoint.pth'.format(round(eval_f1, 4))
                            checkpoint = {
                                "model_state_dict": self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                "epoch": epoch,
                                "scheduler":self.scheduler.state_dict()
                            }
                            torch.save(checkpoint, model_path)
                            print('>> saved: ' + model_path)
                    else:
                        print('eval_f1: {:.4f}, eval_precision: {:.4f}, eval_recall: {:.4f}'.format(eval_f1, eval_precision, eval_recall))
                    print('-' * 100)

        if self.rank == 0:
            model_path = 'last_state_dict/f1_{0}_checkpoint.pth'.format(round(eval_f1, 4))
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                "epoch": epoch,
                "scheduler": self.scheduler
            }
            torch.save(checkpoint, model_path)
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
        good_case = pd.DataFrame(columns=["input","output","ground_truth", "eval_f1"])
        bad_case = pd.DataFrame(columns=["input","output","ground_truth", "eval_f1"])
        other_case = pd.DataFrame(columns=["input","output","ground_truth", "eval_f1"])
        with torch.no_grad():
            with tqdm(total=len(Evalset.dataset.data) // self.args.gpus) as pbar:
                for t_batch, t_sample_batched in enumerate(Evalset):
                    try:
                        # if t_batch > 3:
                        #     break
                        # switch model to training mode, clear gradient accumulators
                        rst = t_sample_batched["rst"].to(self.args.device)
                        input_tensor = t_sample_batched["input_tensor"].to(self.args.device)
                        sent_num = t_sample_batched["sent_num"]
                        reference_tensor = t_sample_batched["reference_tensor"].to(self.args.device)
                        ref_segment_tensor = t_sample_batched["ref_segment_tensor"].to(self.args.device)
                        segment_tensor = t_sample_batched["segment_tensor"].to(self.args.device)
                        # print(t_batch)
                        # if t_batch == 33:
                        #     print(1)
                        with autocast():
                            model_output = self.model(input_tensor, segment_tensor,reference_tensor,ref_segment_tensor, sent_num, rst,mode="eval")
                        encoder_hidden = model_output[0]
                        outputs = []


                        for i in range(len(reference_tensor)):
                            encoder_hidden = model_output[i]
                            # generated = self.beam_search(k=8,encoder = encoder_out)
                            encoder = BaseModelOutput()
                            encoder.last_hidden_state = encoder_hidden
                            generated = self.model(encoder_hidden=encoder, mode="generate")
                            # generated = self.gen(encoder_out)
                            outputs.append(generated)

                        # ground_truth = polarity
                        ground_truth = []
                        for i in range(len(model_output)):
                            ground_truth = reference_tensor[i]
                            model_pred = outputs[i].squeeze(0)
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
                            model_pred = self.args.tokenizer.decode(model_pred).replace('<s>',"").replace('</s>',"")
                            input = self.args.tokenizer.decode(Bart_input_tensor).replace('<s>',"").replace('</s>',"")
                            ground_truth = self.args.tokenizer.decode(ground_truth).replace('<s>',"").replace('</s>',"")
                            res = rouge.get_scores(model_pred, ground_truth)
                            # sari = SARI.SARIsent(input, model_pred, ground_truth)
                            # all_sari.append(sari)
                            eval_f1, eval_test_precision, eval_test_recall = res[0]['rouge-1']['f'], res[0]['rouge-1']['p'], res[0]['rouge-1']['r']
                            if eval_f1 > 0.6:
                                good_case.loc[len(good_case)] = [input, model_pred, ground_truth, eval_f1]
                            elif eval_f1 < 0.3:
                                bad_case.loc[len(bad_case)] = [input, model_pred, ground_truth, eval_f1]
                            else:
                                other_case.loc[len(other_case)] = [input, model_pred, ground_truth, eval_f1]
                            all_f1.append(eval_f1)
                            all_pre.append(eval_test_precision)
                            all_recall.append(eval_test_recall)
                            all_test_correct += n_senti_correct
                        pbar.update(Evalset.batch_size)
                    except Exception as e:
                        print(t_batch)
                        print(e)
                        continue
                # all_test_total += polarity
        # eval_result = torch.argmax(model_pred_all, -1).cpu()
        # ground_truth_all = ground_truth_all.view(-1).cpu()
        # n_correct = (eval_result == ground_truth_all).sum().item()
        good_case.to_excel("samples/good_case.xlsx")
        bad_case.to_excel("samples/bad_case.xlsx")
        other_case.to_excel("samples/other_case.xlsx")
        eval_f1, eval_test_precision, eval_test_recall =sum(all_f1)/len(all_f1) , sum(all_pre)/len(all_pre) , sum(all_recall)/len(all_recall)
        # eval_sari = sum(all_sari)/len(all_sari)
        # print('eval_sari: {:.4f}'.format(eval_sari))
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



    def my_beam_search(self,
    encoder = None,
    batch_size = 1,
    num_beams = 2,
    vocab_size = 50264,
    cur_len = 1,
    embedding_size = 1024,
    hidden_size = 100,
    max_length = 64,
    sos_token_id = 0,
    eos_token_id = 1,
    pad_token_id = 2
                    ):
        beam_scores = torch.zeros((batch_size, num_beams),device=self.device)  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * num_beams, 1), sos_token_id, dtype=torch.long,device=self.device)
        # h0: (1, batch_size * num_beams, hidden_size)
        hidden = torch.zeros((1, batch_size * num_beams, hidden_size))

        while cur_len < max_length:
        # outputs: (batch_size*num_beams, cur_len, vocab_size)


            outputs = self.model(encoder_hidden=encoder, decoder_input_ids=input_ids, mode="generate")
            # 取最后一个timestep的输出 (batch_size*num_beams, vocab_size)
            outputs = outputs.logits
            next_token_logits = outputs[:, -1, :]
            scores = F.log_softmax(next_token_logits, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, num_beams * vocab_size
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // vocab_size  # 1
                    token_id = beam_token_id % vocab_size  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * num_beams + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == num_beams:
                        break
                    # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
            # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break


            # 准备下一次循环(下一层的解码)
            # beam_scores: (num_beams * batch_size)
            # beam_tokens: (num_beams * batch_size)
            # beam_idx: (num_beams * batch_size)
            # 这里beam idx shape不一定为num_beams * batch_size，一般是小于等于
            # 因为有些beam id对应的句子已经解码完了 (下面假设都没解码完)
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
    # 注意有可能到达最大长度后，仍然有些句子没有遇到eos token，这时done[batch_idx]是false
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(num_beams):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
                # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
                # 下面选择若干最好的序列输出
                # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
            # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []
        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        decoded = self.args.tokenizer.decode(decoded.squeeze(0), skip_special_tokens=True)
        return decoded



    def beam_search(self, k, encoder,  max_length = 64):
        hidden_size =self.args.bert_dim
        vocab_size = self.args.vocab_size-1
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
            outputs = self.model(encoder_hidden=encoder, decoder_input_ids=k_prev_words,mode="generate")
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
                complete_seqs.extend(seqs.tolist())  # 加入句子
                complete_seqs_scores.extend(top_k_scores)
                break
            step += 1
        i = complete_seqs_scores.index(max(complete_seqs_scores))  # 寻找score最大的序列
        # 有些许问题，在训练初期一直碰不到eos时，此时complete_seqs为空
        seq = torch.tensor(complete_seqs[i])
        seq = self.args.tokenizer.decode(seq, skip_special_tokens=True)
        return seq
    def gen(self, encoder, decoder_input_ids=None):
        tokenizer = self.args.tokenizer
        device = self.device
        list_v = []
        list_ids = []
        while True:
            # 使用 tokenizer 把输入文本和已经生成的文本转换为模型可以接受的输入格式
            # input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
            encoder_hidden = encoder
            if decoder_input_ids == None:
                decoder_input_ids = torch.tensor([0]).unsqueeze(0).to(device)
                list_ids.append(torch.tensor([0]))
                list_ids.append(torch.tensor([101, 101]))
                list_v.append(torch.tensor([1., 1.]))
                list_v.append(torch.tensor([1., 1.]))

            # 输入到模型中进行生成
            outputs = self.model(encoder_hidden=encoder_hidden, decoder_input_ids=decoder_input_ids)
            # 获取模型的输出
            logits = outputs.logits
            # 获取模型输出的下一个单词的 ID
            next_word_id = logits[0, -1, :].argmax(-1).unsqueeze(0).unsqueeze(0)
            next_words = torch.argmax(logits,dim=2).unsqueeze(0)
            if next_word_id[0][0].item() == 2:
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