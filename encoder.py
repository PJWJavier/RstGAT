# Encoder-Decoder实现英法互译
from __future__ import unicode_literals, print_function, division

import random
import re
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
from torch import optim

# 获取可用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 一句话的开始标志 start of string
SOS_token = 0
# 一句话的结尾标志 end of string
EOS_token = 1


# 要翻译的语言的包装类，包含了常用工具
class Lang:
    def __init__(self, name):
        # 名称
        self.name = name
        # 词语->索引
        self.word2index = {}
        # 词语->计数
        self.word2count = {}
        # 索引->词语
        # 默认添加SOS,EOS
        self.index2word = {0: "SOS", 1: "EOS"}
        # 词语数
        # 因为现在已经有 SOS,EOS 所以=2
        self.n_words = 2  # Count SOS and EOS

    # 添加一句话
    def addSentence(self, sentence):
        # 以空格分割这句话
        # 然后取出每一个词语
        for word in sentence.split(' '):
            # 添加词语
            self.addWord(word)

    def addWord(self, word):
        # 如果以前没有添加过这个词语
        if word not in self.word2index:
            # 索引从0开始
            # 所以先赋值
            # 最后self.n_words+=1
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            # 已经存在则计数+=1
            self.word2count[word] += 1


# 将一个Unicoide编码的字符
# 转换为ASCII编码的字符
# 统一字符编码方便处理
# 将一个Unicode字符串（数据集中的）转换为一个ASCII字符串（输入模型中的）
# 数据标准化
# 一个Unicode字符可以用多种不同的ASCII字符表示
# 转换为统一的形式方便模型处理
def unicodeToAscii(s):
    return ''.join(
        # normalize() 第一个参数指定字符串标准化的方式。
        # NFC表示字符使用单一编码优先，
        # 而NFD表示字符应该分解为多个组合字符表示
        # 先将输入的字符转换
        # 然后再过滤
        # Mn表示Mark
        # 如果不是特殊标记
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# 将字符串规范化
def normalizeString(s):
    # s.lower()先转换为小写
    # .strip()去除首尾的空格
    # 转换为ASCII编码的形式
    s = unicodeToAscii(s.lower().strip())
    # 去除标点符号
    s = re.sub(r"([.!?])", r" \1", s)
    # 去除非字母
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# 从数据集中读取一行数据
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    # Read the file and split into lines
    # 首先以utf-8的方式打开数据集文件
    # read()读取
    # strip()去除多余的空格
    # 以\n分割读取到的内容，也就是分割出每一行
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')
    # Split every line into pairs and normalize
    # 对于数据集中的每一行for l in lines
    # 将每一行以\t分割for s in l.split('\t')
    # 对于分割出来的每一句话s,进行规范化
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # Reverse pairs, make Lang instances
    # 如果要翻转数据集
    # 什么意思呢，就是如果原数据集存放的是英语->法语
    # 如果指定reverse
    # 那么将它进行翻转,变成法语->英语
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


# 一句话的最大长度
MAX_LENGTH = 10
# 选取数据集中的一部分进行训练
# 选取带有一以下前缀的句子
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


# 按照最大长度
# 指定前缀
# 筛选数据集
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and (
            p[0].startswith(eng_prefixes) or p[1].startswith(eng_prefixes))


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# 读取数据集
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))


# 定义Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # 词嵌入
        self.embedding = nn.Embedding(input_size, hidden_size)
        # GRU
        # 因为前面将输入进行了词嵌入，所以输入维度是hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)

    # 前向传递，建立计算图
    def forward(self, input, hidden):
        # 改成[长度，批大小，嵌入维度]的格式
        # 为什么这里长度，批大小都是1呢
        # 因为后面我们是将一句话中的每一个词逐一输入到Encoder中的
        # Decoder同理
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 定义Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 定义带有注意力机制的Decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # attention的输入是词嵌入向量和隐状态
        # 所以输入维度是self.hidden_size*2
        # 因为Decoder输出的句子长度不确定
        # 所以这里输出维度直接取最大了
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 将encoder的输出乘以注意力权重
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 句子->索引
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


# 将句子转换为张量
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# 将数据集中的一个样本转换为张量
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    # 初始化Encoder的隐藏层
    encoder_hidden = encoder.initHidden()
    # 梯度清零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 输入输出的长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    # 将一句话中的每个词语输入到Encoder中
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        # 获取每一步的输出
        encoder_outputs[ei] = encoder_output[0, 0]
    # decoder的输入是一个SOS标记
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # 隐状态是Encoder的最后的隐状态输出
    decoder_hidden = encoder_hidden
    # 是否使用teacher_force的训练模式
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # 如果指定了teacher_force训练模式
        # decoder每一步的输入是真实target中的词语
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        # 指没有指定teacher_force训练模式
        # decoder的下一步的输入是decoder上一步的输出
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            # 如果已经翻译完了
            if decoder_input.item() == EOS_token:
                break
    # 反向传播
    loss.backward()
    # 梯度更新
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


import time
import math


# 秒到分钟转换
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# 获取运行时间间隔
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    # 随机梯度下降优化
    encoder_optimiz.er = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # 获取训练数据
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    # NLLLoss()+LogSoftmax()=CrossEntropy()
    criterion = nn.NLLLoss()
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    showPlot(plot_losses)


import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker


# 画图
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# 模型验证
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
evaluateRandomly(encoder1, attn_decoder1)
output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())


# 可视化注意力
def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")
evaluateAndShowAttention("elle est trop petit .")
evaluateAndShowAttention("je ne crains pas de mourir .")
evaluateAndShowAttention("c est un jeune directeur plein de talent .")