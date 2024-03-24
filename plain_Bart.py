import torch
import os
import torch.nn as nn
import pandas  as pd
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import BartTokenizer
from tqdm import trange, tqdm
from rouge import Rouge
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from configs import GLUE_opts
import torch.multiprocessing as mp
# from Model.Bart_base_Trainer import Trainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast
import torch.distributed as dist
import torch.nn.functional as F
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
class Mydataset(Dataset):
    def __init__(self, fname, tokenizer, max_input_length = 1024):
        data = pd.read_csv(fname)
        all_data = []
        if "train" in fname:
            data = data[:8]
        elif "test" in fname:
            data = data[30050:30058]

        test_samples = data["article"].values
        references = data["abstract"].values
        for i in trange(len(data)):
            inputs = tokenizer(
                test_samples[i],
                padding="max_length",
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt",
            )
            reference = tokenizer(
                references[i],
                padding="max_length",
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            reference_id = reference.input_ids
            reference_mask = reference.attention_mask
            pre_data = {
                "input_ids": input_ids,
                'attention_mask': attention_mask,
                'reference_id': reference_id,
                'reference_mask': reference_mask
            }
            all_data.append(pre_data)
        self.data=all_data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

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

    def train(self, Trainset: DataLoader, Evalset: DataLoader, num_epoch = 10):
        max_test_precision, max_f1, max_recall = 0, 0, 0
        global_step = 0
        scaler = GradScaler()
        for epoch in range(num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0

            with tqdm(total=len(Trainset.dataset.data) // self.args.gpus) as pbar:
                for i_batch, sample_batched in enumerate(Trainset):

                    global_step += 1
                    self.model.train()
                    self.optimizer.zero_grad()
                    input_ids = sample_batched["input_ids"].squeeze().to(self.args.device)
                    attention_mask = sample_batched["attention_mask"].squeeze().to(self.args.device)
                    reference_id = sample_batched["reference_id"].squeeze().to(self.args.device)
                    reference_mask = sample_batched["reference_mask"].squeeze().to(self.args.device)
                    with autocast():
                        outputs = self.model(input_ids, attention_mask=attention_mask)
                        output_id = outputs.logits
                        mask = reference_mask != 0
                        reference_id = torch.masked_select(reference_id, mask)
                        reference_id = torch.roll(reference_id, -1, dims=0)
                        mask = mask.unsqueeze(2).repeat(1, 1, self.args.vocab_size-1)
                        output_id = torch.masked_select(output_id, mask).view(-1, self.args.vocab_size-1)
                        loss = self.criterion(output_id, reference_id.long())
                    # assert model_output.shape[0] == polarity.shape[0]
                    # loss = self.criterion(model_output, polarity.long())
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    n_correct += (torch.argmax(output_id, -1) == reference_id).sum().item()
                    n_total += reference_id.shape[0]
                    pbar.update(Trainset.batch_size)
                senti_train_precision = n_correct / n_total
                print('avg_loss: {:.4f}'.format(loss.item()))
                print('train_precision: {:.4f}'.format(senti_train_precision))
                # switch model to evaluation mode
                self.model.eval()

                eval_precision, eval_f1, eval_recall = self.eval_all(Evalset)
                eval_precision = torch.tensor(eval_precision, dtype=torch.float, device=self.args.device)
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

                if self.args.rank == 0:
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
                            if not os.path.exists('Bart_state_dict'):
                                os.mkdir('Bart_state_dict')
                            model_path = 'Bart_state_dict/f1_{0}_checkpoint.pth'.format(round(eval_f1, 4))
                            checkpoint = {
                                "model_state_dict": self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                "epoch": epoch,
                                "scheduler": self.scheduler.state_dict()
                            }
                            torch.save(checkpoint, model_path)
                            print('>> saved: ' + model_path)
                    else:
                        print('eval_f1: {:.4f}, eval_precision: {:.4f}, eval_recall: {:.4f}'.format(eval_f1, eval_precision,
                                                                                                    eval_recall))
                print('-' * 100)
        model_path = 'Bart_state_dict/f1_{0}'.format(round(eval_f1, 4))
        torch.save(self.model.state_dict(), model_path)
        print('>> saved: ' + model_path)
        return max_test_precision, max_f1, max_recall

    def eval_all(self, Evalset: DataLoader):
        all_test_correct, all_test_total = 0, 0
        ground_truth_all, model_pred_all = None, None
        n_senti_correct = 0
        self.model.eval()
        rouge = Rouge()
        all_f1, all_pre, all_recall = [], [], []
        all_sari = []
        cases = pd.DataFrame(columns=["input", "output", "ground_truth", "eval_f1"])
        with torch.no_grad():
            with tqdm(total=len(Evalset.dataset.data) // self.args.gpus) as pbar:
                for t_batch, t_sample_batched in enumerate(Evalset):
                    try:
                        input_ids = t_sample_batched["input_ids"].squeeze().to(self.args.device)
                        attention_mask = t_sample_batched["attention_mask"].squeeze().to(self.args.device)
                        reference_id = t_sample_batched["reference_id"].squeeze().to(self.args.device)
                        reference_mask = t_sample_batched["reference_mask"].squeeze().to(self.args.device)
                        outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids = reference_id)
                        outputs1 = self.model.module.generate(input_ids, attention_mask=attention_mask,
                                                             repetition_penalty=1.2, num_beams=8)

                        encoder = BaseModelOutput()
                        encoder_hidden = outputs.encoder_last_hidden_state
                        encoder.last_hidden_state = encoder_hidden
                        # encoder = outputs.encoder_last_hidden_state
                        # outputs1 = self.model(decoder_input_ids = reference_id,attention_mask=attention_mask, encoder_outputs = (encoder, None, None))
                        # outputs1 = self.my_beam_search(encoder=encoder,attention_mask=attention_mask,batch_size=self.args.batch_size,num_beams=8)

                        out = self.model.module.generate(input_ids = None,attention_mask=attention_mask,
                                                         repetition_penalty=1.2,num_beams=8,encoder_outputs=encoder)

                        input_all = self.args.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                        output_str_all = self.args.tokenizer.batch_decode(out, skip_special_tokens=True)
                        output_str_all1 = self.args.tokenizer.batch_decode(outputs1, skip_special_tokens=True)
                        reference_all = self.args.tokenizer.batch_decode(reference_id, skip_special_tokens=True)
                        for i in range(self.args.batch_size):
                            input = input_all[i]
                            output_str = output_str_all[i]
                            reference = reference_all[i]
                            res = rouge.get_scores(output_str, reference)
                            eval_f1, eval_test_precision, eval_test_recall = res[0]['rouge-l']['f'], \
                                                                             res[0]['rouge-l']['p'], \
                                                                             res[0]['rouge-l']['r']
                            cases.loc[len(cases)] = [input, output_str, reference, eval_f1]

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
            cases.to_excel("Bart_samples/cases.xlsx")
            eval_f1, eval_test_precision, eval_test_recall = sum(all_f1) / len(all_f1), sum(all_pre) / len(
                all_pre), sum(all_recall) / len(all_recall)
            return eval_test_precision, eval_f1, eval_test_recall

    def eval(self, Evalset: DataLoader):
        all_test_correct, all_test_total = 0, 0
        ground_truth_all, model_pred_all = None, None
        n_senti_correct = 0
        self.model.eval()
        rouge = Rouge()
        all_f1, all_pre, all_recall = [], [], []
        all_sari = []
        good_case = pd.DataFrame(columns=["input", "output", "ground_truth", "eval_f1"])
        bad_case = pd.DataFrame(columns=["input", "output", "ground_truth", "eval_f1"])
        other_case = pd.DataFrame(columns=["input", "output", "ground_truth", "eval_f1"])
        with torch.no_grad():
            with tqdm(total=len(Evalset.dataset.data)  // self.args.gpus) as pbar:
                for t_batch, t_sample_batched in enumerate(Evalset):
                    try:
                        input_ids = t_sample_batched["input_ids"].squeeze().to(self.args.device)
                        attention_mask = t_sample_batched["attention_mask"].squeeze().to(self.args.device)
                        reference_id = t_sample_batched["reference_id"].squeeze().to(self.args.device)
                        reference_mask = t_sample_batched["reference_mask"].squeeze().to(self.args.device)
                        outputs = self.model.module.generate(input_ids, attention_mask=attention_mask)
                        input_all = self.args.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                        output_str_all = self.args.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        reference_all = self.args.tokenizer.batch_decode(reference_id, skip_special_tokens=True)
                        for i in range(self.args.batch_size):
                            input = input_all[i]
                            output_str = output_str_all[i]
                            reference = reference_all[i]
                            res = rouge.get_scores(output_str, reference)
                            eval_f1, eval_test_precision, eval_test_recall = res[0]['rouge-l']['f'], res[0]['rouge-l']['p'], \
                                                                             res[0]['rouge-l']['r']
                            if eval_test_precision > 0.6:
                                good_case.loc[len(good_case)] = [input, output_str, reference, eval_f1]
                            elif eval_test_precision < 0.3:
                                bad_case.loc[len(bad_case)] = [input, output_str, reference, eval_f1]
                            else:
                                other_case.loc[len(other_case)] = [input, output_str, reference, eval_f1]
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
            good_case.to_excel("Bart_samples/good_case.xlsx")
            bad_case.to_excel("Bart_samples/bad_case.xlsx")
            other_case.to_excel("Bart_samples/other_case.xlsx")
            eval_f1, eval_test_precision, eval_test_recall = sum(all_f1) / len(all_f1), sum(all_pre) / len(
                all_pre), sum(all_recall) / len(all_recall)
            return eval_test_precision, eval_f1, eval_test_recall

    def my_beam_search(self,
    encoder = None,
    attention_mask = None,
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
        attention_mask = attention_mask.repeat(num_beams,1)
        encoder =  encoder.repeat(num_beams,1,1)
        while cur_len < max_length:
        # outputs: (batch_size*num_beams, cur_len, vocab_size)


            outputs = self.model(decoder_input_ids = input_ids,attention_mask=attention_mask, encoder_outputs = (encoder, None, None))
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
        # decoded = self.args.tokenizer.decode(decoded.squeeze(0), skip_special_tokens=True)
        return decoded





import argparse
from Model.Utils import _get_total_training_steps, print_args
def main(gpu, args):
    fname = 'Data/efact/'+ 'test_rst_end.csv'
    rank = gpu
    args.rank = gpu
    device = "cuda:" + str(gpu)
    device = torch.device(device)
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    args.device = device
    batch_size = 2
    args.batch_size = batch_size
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn',cache_dir='./BART_large',local_files_only=True)
    args.tokenizer = tokenizer
    Trainset = Mydataset('Data/efact/'+ 'train_rst_end.csv', tokenizer)
    Evalset = Mydataset('Data/efact/'+ 'test_rst_end.csv', tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        Trainset,
        num_replicas=args.world_size,
        rank=rank
    )
    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        Evalset,
        num_replicas=args.world_size,
        rank=rank
    )
    TrainLoader = DataLoader(dataset=Trainset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             pin_memory=True, sampler=train_sampler)
    EvalLoader = DataLoader(dataset=Evalset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            pin_memory=True, sampler=eval_sampler)
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn',cache_dir='./BART_large',local_files_only=True).to(device)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    args.vocab_size = tokenizer.vocab_size
    criterion = nn.CrossEntropyLoss()
    _params = model.parameters()
    optimizer = args.optimizer(_params, lr=args.learning_rate, weight_decay=args.l2reg, correct_bias=True)
    num_warmup_steps = 100
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, _get_total_training_steps(args, Trainset))
    if args.continue_training:
        checkpoint = torch.load('Bart_state_dict/f1_0.3838_checkpoint.pth',map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("checkpoint loaded")


    trainer     = Trainer(args, model, criterion, optimizer, scheduler, device)
    max_test_precision, max_f1, max_recall = trainer.train(TrainLoader, EvalLoader)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    GLUE_opts(parser)
    args = parser.parse_args()
    args.optimizer = AdamW
    args.world_size = args.gpus * args.nodes  #
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    args.continue_training = True
    mp.spawn(main, nprocs=args.gpus, args=(args,))


