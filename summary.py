import torch
import os
import torch.nn as nn
import pandas  as pd
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import pipeline
from tqdm import trange
from rouge import Rouge

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
## Setting to use the bart-large-cnn model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
rouge = Rouge()

# model = BartForConditionalGeneration.from_pretrained('facebook/bart-base',cache_dir='./BART',local_files_only=True)
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',cache_dir='./BART',local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base',cache_dir='./BART',local_files_only=True)
max_input_length = 1024
fname = 'Data/efact/'+ 'test_rst_end.csv'
data = pd.read_csv(fname)
data = data[2000:2500]
test_samples = data["article"].values
ground_truth = data["abstract"].values
all_f1, all_pre, all_recall = [], [], []
good_case = pd.DataFrame(columns=["input","output","ground_truth"])
bad_case = pd.DataFrame(columns=["input","output","ground_truth"])
other_case = pd.DataFrame(columns=["input","output","ground_truth"])
for i in trange(len(test_samples)):

    summary_text = summarizer(test_samples[i], max_length=100, min_length=5, do_sample=False)[0]['summary_text']

    res = rouge.get_scores(summary_text, ground_truth[i])
    eval_f1, eval_test_precision, eval_test_recall = res[0]['rouge-1']['f'], res[0]['rouge-1']['p'], res[0]['rouge-1'][
        'r']
    if eval_test_precision > 0.6:
        good_case.loc[len(good_case)] = [test_samples[i], output_str[0], ground_truth[i]]
    elif eval_test_precision < 0.3:
        bad_case.loc[len(bad_case)] = [test_samples[i], output_str[0], ground_truth[i]]
    else:
        other_case.loc[len(other_case)] = [test_samples[i], output_str[0], ground_truth[i]]
    all_f1.append(eval_f1)
    all_pre.append(eval_test_precision)
    all_recall.append(eval_test_recall)
good_case.to_excel("samples/good_case.xlsx")
bad_case.to_excel("samples/bad_case.xlsx")
other_case.to_excel("samples/other_case.xlsx")
eval_f1, eval_test_precision, eval_test_recall =sum(all_f1)/len(all_f1) , sum(all_pre)/len(all_pre) , sum(all_recall)/len(all_recall)
print('eval_f1: {:.4f}, eval_precision: {:.4f}, eval_recall: {:.4f}'.format(eval_f1, eval_test_precision, eval_test_recall))


