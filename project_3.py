import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense,Bidirectional
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow.keras.utils import to_categorical
import pandas as pd
import codecs
import jieba
import numpy as np
import os
from string import digits


class DataClean:
    '''
        此类主要讲数据多个标签和多分类合并成20*4的矩阵,并将原分类数据11映射到新的矩阵中.
        由于测试集为空,故只能验证集作为测试集合一并处理
    '''
    def __init__(self,train_data_path,test_data_path,stop_text_path):
        '''
        :param train_data_path:训练数据文件地址
        :param test_data_path: 测试数据文件地址
        :param stop_text_path: 停词文件地址
        '''
        self.__train_data=pd.read_csv(train_data_path)
        self.__test_data=pd.read_csv(test_data_path)
        # 提取和处理停词
        self.__stop_text=codecs.open(stop_text_path,'r',encoding='utf-8')
        self.__stop_words={}.fromkeys([line.strip() for line in self.__stop_text])
        self.__stop_text.close()

    def _cut_word(self,sen):
        '''

        :param sen: 需要切词的文本向量
        :return:切词结束的列表
        '''
        result=jieba.lcut(sen)
        cut_result=[x for x in result if x not in self.__stop_words]
        result=' '.join(cut_result)
        return result

    def _replace_n(self,x):
        x=x.replace('\n','')
        return x

    def _cut_num(self,x):
        '''
        去除文字中的纯数字
        :param x: 需要被切除数字的字符串
        :return: 切除后的字符串
        '''
        remove_role = str.maketrans('', '', digits)
        x=x.translate(remove_role)
        return x

    def main(self,save_new_train_data_path=None,save_new_test_data_path=None):
        '''
        :return:处理完切新生成的训练集/测试集矩阵和停词字典
        '''
        if not len(self.__train_data):
            raise ValueError('文本读取错误!')
        labels={1:'-Positive',0:'-Middle',-1:'-Negative',-2:'-NoComment'}
        # 处理字符串中特殊换行符和纯数字
        self.__train_data.content=self.__train_data.content.apply(self._cut_word)
        self.__train_data.content=self.__train_data.content.apply(self._cut_word)
        self.__train_data.content = self.__train_data.content.apply(self._cut_num)


        self.__test_data.content=self.__test_data.content.apply(self._cut_word)
        self.__test_data.content = self.__test_data.content.apply(self._cut_num)
        self.__test_data.content=self.__test_data.content.apply(self._cut_word)

        new_col=[x+y for x in self.__train_data.columns[2:] for y in list(labels.values())]
        new_train_data=pd.DataFrame(np.zeros((self.__train_data.shape[0],len(new_col))),columns=new_col)
        new_test_data=pd.DataFrame(np.zeros((self.__test_data.shape[0],len(new_col))),columns=new_col)
        # 替换成nan值
        new_train_data.replace(0,3,inplace=True)
        new_test_data.replace(0,3,inplace=True)
        for (train_index,train_items),(test_index,test_items) in zip(self.__train_data.iterrows(),self.__test_data.iterrows()):
            for x_train_index in train_items.index[2:]:
                for y in list(labels.keys()):
                    if train_items[x_train_index]==y:
                        new_train_data.loc[train_index,x_train_index+labels[y]]=y
                    else:
                        continue

            for x_test_index in test_items.index[2:]:
                for y in list(labels.keys()):
                    if test_items[x_test_index]==y:
                        new_test_data.loc[test_index,x_test_index+labels[y]]=y
                    else:
                        continue
        new_train_data['content']=self.__train_data.content
        new_test_data['content']=self.__test_data.content
        # 如果存储地址不为空,就存到csv下
        if save_new_test_data_path and save_new_train_data_path:
            new_test_data.to_csv(save_new_train_data_path,encoding='gb18030')
            new_train_data.to_csv(save_new_test_data_path,encoding='gb18030')

        self.__train_data=new_train_data
        self.__test_data=new_test_data

        return self.__train_data,self.__test_data,self.__stop_words

class MyLSTM:
    def __init__(self,train_data_path,test_data_path,vocal_evct_num=6000,sentence_max_len=100):
        '''
        :param train_data_path: 处理过后训练集地址
        :param test_data_path: 处理过后测试集地址
        :param vocal_num: 需要被向量化单词的个数
        :param sentence_max_len: 对于标签需要padding到句子长度
        '''
        self.__train_data_path=train_data_path
        self.__test_data_path=test_data_path
        self.__vocal_evct_num=vocal_evct_num
        self.__sentence_max_len=sentence_max_len

    def _regularising_data(self):
        if self.__train_data_path is None or self.__test_data_path is None:
            raise ValueError('两个文件地址缺一不可!')

        train=pd.read_csv(self.__train_data_path,engine='python',encoding='gb18030').dropna()
        test=pd.read_csv(self.__test_data_path,engine='python',encoding='gb18030').dropna()
        train['labels']=[np.nan]*train.shape[0]
        test['labels']=[np.nan]*test.shape[0]

        self.__train_data=train.iloc[:,-2].to_numpy()
        self.__test_data=test.iloc[:,-2].to_numpy()
        i=0
        for index1,items1 in train.iterrows():
            local_train=items1.replace([3,-2],np.nan)
            local_train=local_train.iloc[1:-2].dropna().sum()
            train.iloc[i,-1]=np.where(local_train>0,1.0,-1.0)
            i+=1
        i=0
        for index2,items2 in test.iterrows():
            local_test=items2.replace([3,-2],np.nan)
            local_test=local_test.iloc[1:-2].dropna().sum()
            test.iloc[i, -1] = np.where(local_test > 0, 1.0, -1.0)
            i+=1
        self.__train_labels=np.reshape(train.iloc[:,-1],-1).to_numpy()
        self.__test_labels=np.reshape(test.iloc[:,-1],-1).to_numpy()


    def _vectorised_trained_and_test_words(self):
        self.__tokeniser = Tokenizer(num_words=self.__vocal_evct_num)
        self.__tokeniser.fit_on_texts(self.__train_data)

        self.__train_data=self.__tokeniser.texts_to_sequences(self.__train_data)
        self.__train_data=sequence.pad_sequences(self.__train_data,maxlen=self.__sentence_max_len)
        self.__test_data=self.__tokeniser.texts_to_sequences(self.__test_data)
        self.__test_data=sequence.pad_sequences(self.__test_data,maxlen=self.__sentence_max_len)
        return self.__train_data,self.__test_data

    def _build_layers_and_train_model(self):
        inputs=keras.Input(shape=[self.__sentence_max_len])
        x=Embedding(self.__vocal_evct_num,256,input_length=self.__sentence_max_len)(inputs)
        x=Bidirectional(LSTM(256,implementation=2))(x)
        x=Dense(256,activation='relu')(x)
        x=Dropout(0.5)(x)
        outputs=Dense(10,activation='softmax')(x)
        model = keras.Model(inputs, outputs, name='Multi-LSTM')

        print(model.summary())
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


    def _fit_and_scoring_model(self,model,vectorised_trained_words,vectorised_test_words,
                               model_save_path='/home/abc/下载/jarek/NLP_assessment/Project3/'):
        fitted_model=model.fit(vectorised_trained_words[25000:],self.__train_labels[25000:],epochs=10,
                               validation_data=(vectorised_trained_words[:25000],self.__train_labels[:25000]))
        fitted_model.save('./LSTM_model.h5')
        score=fitted_model.evaluate(vectorised_test_words,self.__test_laels)
        return score

    def main(self):
        self._regularising_data()
        vectorised_trained_words, vectorised_test_words=self._vectorised_trained_and_test_words()
        model=self._build_layers_and_train_model()
        score=self._fit_and_scoring_model(model,vectorised_trained_words, vectorised_test_words)
        print(score)

if __name__=='__main__':
    # train_data_path='/home/abc/下载/jarek/NLP_assessment/Project3/comment-classification/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
    # test_data_path='/home/abc/下载/jarek/NLP_assessment/Project3/comment-classification/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
    # stop_text_path='/home/abc/下载/jarek/NLP_assessment/Project3/stop_words.txt'
    # model=DataClean(train_data_path,test_data_path,stop_text_path)
    # new_train_data,new_test_data,stop_words=model.main('/home/abc/下载/jarek/NLP_assessment/Project3/new_test_data.csv',
    #                                                    '/home/abc/下载/jarek/NLP_assessment/Project3/new_train_data.csv')
    myLSTM=MyLSTM('/home/abc/下载/jarek/NLP_assessment/Project3/new_train_data.csv',
                  '/home/abc/下载/jarek/NLP_assessment/Project3/new_test_data.csv')
    score=myLSTM.main()