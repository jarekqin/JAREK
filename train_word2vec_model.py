#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:26:20 2019

@author: abc
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:26:20 2019

@author: abc
"""

# word2vec
# 所有语料已经经过繁体－>简体的转换
import jieba
import gensim
import codecs, sys
import logging
import multiprocessing
import codecs
from tqdm import tqdm
import os

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CleanData:
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                if len(line) > 0:
                    yield [segment.strip() for segment in jieba.cut(line.strip(), cut_all=False)
                           if segment not in stoplist and len(segment) > 0]


def is_other(instr):
    out_str = ''
    for index in range(len(instr)):
        if is_chinese(instr[index]):
            out_str = out_str + instr[index].strip()
    return out_str


def is_chinese(uchar):
    # """判断一个unicode是否是汉字"""
    if u'\u4e00' <= uchar <= u'\u9fff':
        return True



def tsne_plot_all_words(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

def plot_one_word_similar_words(word,number,model):
    closed_word=model.most_similar(word,topn=number)
    labels = []
    tokens = []

    for word in range(len(closed_word)):
        tokens.append(closed_word[word][0])
        labels.append(model[closed_word[word][0]])

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(labels)

    x, y = [], []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 12))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(tokens[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

if __name__ == '__main__':
    # print('开始获取语料和进行中文切词')
    # dirname = 'chinese_simplified'
    # sentences = CleanData(dirname)
    # print('第一步获取结束')
    # print('-' * 20, '华丽的分割线', '-' * 20)
    # print('开始清洗数据')
    # # 读取停用词；
    # stop_f = codecs.open(u'stop_words.txt', 'r', encoding='utf-8')
    # stoplist = {}.fromkeys([line.strip() for line in stop_f])
    # # 分词结果写入文件
    # f = codecs.open('project_wiki_003.txt', 'w', encoding='utf-8')
    # i = 0
    # j = 0
    # w = tqdm(sentences, desc=u'分词句子')
    # for sentence in w:
    #     if len(sentence) > 0:
    #         output = " "
    #         for d in sentence:
    #             # 去除停用词；
    #             if d not in stoplist:
    #                 output += is_other(d).strip() + " "
    #         f.write(output.strip())
    #         f.write('\r\n')
    #         i += 1
    #         if i % 10000 == 0:
    #             j += 1
    #             w.set_description(u'已分词： %s万个句子' % j)
    # f.close()
    # print()
    # print('清洗结束')
    # print('-' * 20, '华丽的分割线', '-' * 20)
    # print('开始训练模型')
    # program = 'word2vec_model.py'
    # logger = logging.getLogger(program)
    #
    # logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    # logging.root.setLevel(level=logging.INFO)
    # logger.info("running %s" % ' '.join(program))

    # infile = 'project_wiki_003.txt'
    # vec_outfile1 = 'project_wiki.zh.text.model'
    # vec_outfile2 = 'project_wiki.zh.text.vector'
    # sentences = LineSentence(infile)
    #
    # model = Word2Vec(LineSentence(infile), size=300, window=20, min_count=5,
    #                  workers=multiprocessing.cpu_count())
    #
    # model.save(vec_outfile1)
    # model.wv.save_word2vec_format(vec_outfile2, binary=False)
    # print('训练结束')
    # print('-' * 20, '华丽的分割线', '-' * 20)


    model=Word2Vec.load('project_wiki.zh.text.model')
    word1='人物'
    word2='美国'
    word3='科学家'

    # 找出对应的词向量
    for word in [word1,word2,word3]:
        if word in model:
            print('%s的词向量是:' % word)
            print(model[word])
        else:
            print('该单词不存在词向量!')

    # 找出近义词
    result={}
    for word in [word1,word2,word3]:
        result[word]=model.most_similar(word)
        print('与%s相似度最好的近义词是:' % word)
        print('-' * 20, '华丽的分割线', '-' * 20)
        for ans in result[word]:
            print(ans)
            print('-' * 20, '华丽的分割线', '-' * 20)

    # 判断每个单词之间的相似度
    print('%s和%s的相似度是:%0.5f' % (word1,word2,model.similarity(word1,word2)))
    print('-' * 20, '华丽的分割线', '-' * 20)
    print('%s和%s的相似度是:%0.5f' % (word1,word3,model.similarity(word1,word3)))
    print('-' * 20, '华丽的分割线', '-' * 20)
    print('%s和%s的相似度是:%0.5f' % (word2,word3,model.similarity(word2,word3)))
    print('-' * 20, '华丽的分割线', '-' * 20)



    # 找出三个单词的离群词
    print('3个单词的离群词有-->',model.doesnt_match([word1,word2,word3]))
    print('-' * 20, '华丽的分割线', '-' * 20)

    # 最后案例测试
    print('和美国最相近的100个词语是:')
    ans=(model.most_similar(['美国'],topn=100))
    for i in range(len(ans)):
        print('%s: %s' % (ans[i][0],ans[i][1]))
        print('-' * 20, '华丽的分割线', '-' * 20)

    plot_one_word_similar_words(['中国'],100,model)
    #
    # model = Word2Vec(LineSentence(infile), size=100, window=20, min_count=500,
    #                  workers=multiprocessing.cpu_count())
    # tsne_plot(model)

    # 测试单个单词和画出最近单词图



