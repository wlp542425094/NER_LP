
# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)




def log_sum_exp(vec):   #vec是1*5, type是Variable
    max_score = vec[0, argmax(vec)]
    # max_score维度是１，　max_score.view(1,-1)维度是１＊１，
    # max_score.view(1, -1).expand(1, vec.size()[1])的维度是１＊５
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])

    # 里面先做减法，减去最大值可以避免e的指数次，计算机上溢
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))    #等价于return torch.log(torch.sum(torch.exp(vec)))

class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim//2,num_layers=1,bidirectional=True)

        # Maps the outputs into LSTM into tag space
        self.hidden2tag = nn.Linear(hidden_dim,self.tagset_size)

        # Matrix of trainision parameters. emtry i.j is the score of transitionsing to j from j
        self.transitions = nn.Parameter(torch.randn(self.tagset_size,self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    # 预测序列的得分
    # 只是根据随机的transitions，前向传播算出的一个score
    # 用到了动态规划的思想，但因为用的是随机的转移矩阵，算出的值很大score>20
    def _forward_alg(self, feats):

        init_alphas = torch.full((1,self.tagset_size),-10000) # 1*5 而且全是-10000

        # START_TAG has all of the score
        # 因为start tag是4，所以tensor([[-10000., -10000., -10000., 0., -10000.]])，
        # 将start的值为零，表示开始进行网络的传播，
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_alphas   # 初始状态的forward_var，随着step t变化
        # 会迭代feats的行数次
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep #feat的维度是５ 依次把每一行取出来~
            for next_tag in range(self.tagset_size):  #next tag 就是简单 i，从0到len
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)  # 维度是1*5 LSTM生成的矩阵是emit score
                trans_score = self.transitions[next_tag].view(1, -1)  # 维度是1*5

                # 第一次迭代时理解：
                # trans_score所有其他标签到Ｂ标签的概率
                # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
                # 此时的alphas t 是一个长度为5，例如<class 'list'>:
                # [tensor(0.8259), tensor(2.1739), tensor(1.3526), tensor(-9999.7168), tensor(-0.7102)]
            forward_var = torch.cat(alphas_t).view(1, -1)

        # 最后只将最后一个单词的forward var与转移 stop tag的概率相加
        # tensor([[   21.1036,    18.8673,    20.7906, -9982.2734, -9980.3135]])
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)   # alpha是一个0维的tensor
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags): #feats 11*5  tag 11 维
        score = torch.zeros(1)
        # 将START_TAG的标签３拼接到tag序列最前面，这样tag就是12个了
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i 的输出结果，有５个值，
            # 对应B, I, E, START_TAG, END_TAG, 取对应标签的值
            # transition【j,i】 就是从i ->j 的转移概率值
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # initialize the viterbi variables in long space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = [] # holds the backpointers for this step
            viterbivars_t = [] # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next-tag_var[i] holds the viterbi variable for tag i
                # at the previous step, plus the score of transitioning
                # from tag i to next_tag.
                # we don't include the emission scores here because the max
                # does not depend on them(we add them in below)
                # 其他标签（B,I,E,Start,End）到标签next_tag的概率
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # now add in the emssion scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # 从step0到step(i-1)时5个序列中每个序列的最大score
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t) # bptrs_t有５个元素

        # transition to STOP_TAG
        # 其他标签到STOP_TAG的转移概率
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # follow the back pointers to decode the best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # pop off the start tag
        # we don't want to return that ti the caller
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG] # Sanity check
        best_path.reverse() # 把从后向前的路径正过来
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # feats: 11*5 经过了LSTM+Linear矩阵后的输出，之后作为CRF的输入。
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score


    def forward(self, sentence):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)


        return score, tag_seq
