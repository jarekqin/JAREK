import pandas as pd
import numpy as np
import jieba
from typing import List,Dict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


class MyKnn_Verify_Copy_News:
    def __init__(self,
                 data_path:str='/home/abc/下载/NLP_assessment/lesson7+assessment7/sqlResult_1558435.csv',
                 model_type='nb'):
        self.__data_path=data_path
        self.__model_type=model_type

    # def __is_chinese(self,uchar:str):
    #     # """判断一个unicode是否是汉字"""
    #     if u'\u4e00' <= uchar <= u'\u9fff':
    #         return True

    # def __clean_line(self,s:str)->str:
    #     """
    #     :param s: 清洗爬取的中文语料格式
    #     :return:
    #     """
    #     import re
    #     from string import digits, punctuation
    #     rule = re.compile(u'[^a-zA-Z.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：' + digits + punctuation + '\u4e00-\u9fa5]+')
    #     s = re.sub(rule, '', s)
    #     s = re.sub('[、]+', '，', s)
    #     s = re.sub('\'', '', s)
    #     s = re.sub('[#]+', '，', s)
    #     s = re.sub('[?]+', '？', s)
    #     s = re.sub('[;]+', '，', s)
    #     s = re.sub('[,]+', '，', s)
    #     s = re.sub('[!]+', '！', s)
    #     s = re.sub('[.]+', '.', s)
    #     s = re.sub('[，]+', '，', s)
    #     s = re.sub('[。]+', '。', s)
    #     s = s.strip().lower()
    #     return s

#    def __cut(self,string):
#        return ' '.join(jieba.cut(string))

    def __clean_data(self)->pd.DataFrame:
        print('开始下载和清洗数据')
        self.__data = pd.read_csv(self.__data_path,encoding='gb18030')
        self.__data=self.__data.fillna('')
        self.__data['content']=self.__data['content'].apply(lambda x:' '.join(jieba.cut(x)))
        self.__data['news_labels']=np.where(self.__data['source'].str.contains('新华社'), 1, 0)
        ratio=len(self.__data[self.__data['news_labels']==1])/ \
               len(self.__data)
        print('取得的新华社新闻占比为%0.5f%s' %
              (ratio*100,'%'))


    def get_original_labels(self)->pd.Series:
        return self.__data['news_labels']

    # 获得角度矩阵和拆分数据集
    def __TfidfVectorizer(self)->List[str]:
        print('我们有数据%d行%d列'%(self.__data.shape[0],self.__data.shape[1]))
        train_size=int(input('输入训练集大小(最大推荐为50000): '))
        if not train_size:
            raise ('训练集大小不能为空')
        max_f=int(input('输入最大特征数量(推荐最大为3000): '))
        if not max_f:
            raise('特征最大数量不能为空')
        vectorizer = TfidfVectorizer(max_features=max_f)
        total_data=self.__data['content'].values[:train_size]
        total_labels=self.__data['news_labels'].values[:train_size]
        total_data=vectorizer.fit_transform(total_data)
        print('我们得到了以下大小的词向量角度矩阵:')
        print(total_data.shape)
        test_size=float(input('输入想要拆分数据集的百分比:'))
        x_train,x_test,y_train,y_test=train_test_split(total_data,total_labels,
                                                       test_size=test_size,
                                                       random_state=0)
        return x_train,x_test,y_train,y_test,total_data,total_labels

    # nb+dt
    def __training_model(self,model_type,x_train:List[float],y_train:List[int],weight:List[int]=None,
                         alpha:float=0.01,binarize=0.0):
        if model_type=='nb':
            if len(weight)==0 and not isinstance(weight,list):
                model=GaussianNB()
                model.fit(x_train.toarray(),y_train)
            else:
                model=GaussianNB(priors=weight)
                model.fit(x_train.toarray(),y_train)
        elif model_type=='bnb':
            # 不设定过大特征个数
            model=BernoulliNB(alpha=alpha,binarize=binarize,
                              fit_prior=True,class_prior=None)
            model.fit(x_train.toarray(),y_train)
        else:
            raise('错误的模型')
        return model

    def __predict(self,model,x_,y_):
        prediction=model.predict(x_.toarray())
        accurate=model.score(x_.toarray(),y_)
        print('得分是%0.5f' % accurate)
        return prediction


    # 混交矩阵和三种指标
    def __performance_indicator(self,model,
                                x_train:List[float],
                                y_train:List[int],
                                sample=1000):
        precision=None
        recall=None
        roc_aoc=None
        confusion_max=None
        random_set = np.random.choice(np.arange(len(y_train)), sample)
        sub_x = x_train[random_set]
        sub_y = y_train[random_set]
        y_hat = model.predict(sub_x.toarray())
        model_score=model.score(sub_x.toarray(),sub_y)
        print('准确率: {}'.format(precision_score(sub_y, y_hat)))
        precision=precision_score(sub_y, y_hat)
        print('召回率: {}'.format(recall_score(sub_y, y_hat)))
        recall=recall_score(sub_y, y_hat)
        print('ROC_AOC曲线: {}'.format(roc_auc_score(sub_y, y_hat)))
        roc_aoc=roc_auc_score(sub_y, y_hat)
        print('混肴矩阵: \n{}'.format(confusion_matrix(sub_y, y_hat, labels=[0, 1])))
        confusion_max=confusion_matrix(sub_y, y_hat, labels=[0, 1])
        print('模型打分: {}'.format(model_score))
        return [precision,recall,roc_aoc,confusion_max,model_score]

    def verify_atricle(self,prediction_labels:List[int])->float:
        j=1
        for i in range(len(self.__data)):
            if (prediction_labels[i]==1) and (self.__data.iloc[i]['news_labels']==0) and\
                    self.__data.iloc[i]['content']:
                print('来源:{0},内容是{1},被判断是抄袭新华社的消息' .format(
                    self.__data.iloc[i]['source'],
                    self.__data.iloc[i]['content']
                ))
                j+=1
            else:
                continue
        print('一共%d条新闻被认定是抄袭新华社'%j)
        ratio=j/len(self.__data)
        print('判别出的抄袭新闻占总新闻的%0.2f%s' % (ratio*100,'%'))
        return ratio*100

    def main(self):
        self.__clean_data()
        print('数据清洗完毕')
        x_train,x_test,y_train,\
        y_test ,total_data,total_labels=self.__TfidfVectorizer()
        if self.__model_type=='nb':
            print('开始训练朴素贝叶斯模型')
            # 用普通朴素贝叶斯模型形成的结果
            fitted_model=self.__training_model('nb',x_train,y_train,weight=[0.5,0.5])
            print('模型训练完毕')
            print('-'*40)
            print('进行模型准确率->ROC_AOC曲线表现打分')
            # 通过选取不同大小的随机样本集进行测试
            training_score=[]
            testing_score=[]
            i=1
            for sample in range(1000,10000,1000):
                print('第%d批次训练集准确率->ROC_AOC曲线打分'%i)
                training_score.append(
                    ('第%d批次训练集准确率->ROC_AOC曲线打分'%i,sample,
                     self.__performance_indicator(fitted_model,x_train,y_train,sample)))
                print('-'*40)
                print('第%d批次测试集准确率->ROC_AOC曲线打分'%i)
                self.__performance_indicator(fitted_model,x_test,y_test,sample)
                testing_score.append(
                    ('第%d批次测试集准确率->ROC_AOC曲线打分'%i,sample,
                     self.__performance_indicator(fitted_model,x_test,y_test,sample)))
                print('-'*40)
                i+=1
            print('模型打分结束')
            print('-'*40)
            print('对固定训练集和测试集进行预测开始')
            print('固定训练集预测和打分')
            train_predictions=self.__predict(fitted_model,x_train,y_train)
            print('固定测试集预测和打分')
            test_predictions=self.__predict(fitted_model,x_test,y_test)
            print('-'*40)
            print('对全体文章进行预测')
            total_predcitons=self.__predict(fitted_model,total_data,total_labels)
            print('普通朴素贝叶斯模型结束')
            return training_score, testing_score,total_predcitons


        elif self.__model_type=='bnb':
            print('-'*40)
            print('伯努利贝叶斯模型开始')
            print('开始训练模型')
            models=[]
            for part_alpha,part_binarize in zip(np.linspace(0.1,1.0,10),np.linspace(0,10,10)):
                fitted_model = self.__training_model('bnb', x_train, y_train,
                                                     alpha=part_alpha,
                                                     binarize=part_binarize)

                print('alpha:%0.2f,binarize:%d的模型训练完毕' % (part_alpha,part_binarize))
                models.append(fitted_model)
                print('-' * 40)
                print('进行模型准确率->ROC_AOC曲线表现打分')
                # 通过选取不同大小的随机样本集进行测试
                training_score = []
                testing_score = []
                i = 1
                for sample in range(1000, 10000, 1000):
                    print('第%d批次训练集准确率->ROC_AOC曲线打分' % i)
                    training_score.append(
                        ('第%d批次训练集准确率->ROC_AOC曲线打分' % i, i,sample,part_alpha,part_binarize,
                         self.__performance_indicator(fitted_model, x_train, y_train, sample)))
                    print('-' * 40)
                    print('第%d批次测试集准确率->ROC_AOC曲线打分' % i)
                    self.__performance_indicator(fitted_model, x_test, y_test, sample)
                    testing_score.append(
                        ('第%d批次测试集准确率->ROC_AOC曲线打分' % i,i, sample,sample,part_alpha,part_binarize,
                         self.__performance_indicator(fitted_model, x_test, y_test, sample)))
                    i += 1
                print('模型打分结束')
                print('-' * 40)
            print('选择模型打分最高的模型')
            highest_score_list=sorted(training_score,key=lambda x:x[:][-1],reverse=True)[0]
            fitted_model=models[highest_score_list[1]]
            print('选择模型打分最高的模型')
            print('-' * 40)
            print('对训练集和测试集进行预测开始')
            print('训练集预测和打分')
            train_predictions = self.__predict(fitted_model, x_train, y_train)
            print('测试集预测和打分')
            test_predictions = self.__predict(fitted_model, x_test, y_test)
            print('对全体文章进行预测')
            total_predcitons = self.__predict(fitted_model, total_data, total_labels)
            return training_score, testing_score,total_predcitons

        else:
            raise('错误的模型输入')






if __name__=='__main__':
    print('朴素贝叶斯开始')
    # 朴素贝叶斯模型
    bn_model=MyKnn_Verify_Copy_News(model_type='nb')
    training_score, testing_score,total_predcitons=bn_model.main()
    # 获取处理过后的全数据
    original_labels=bn_model.get_original_labels()
    # 输出被判定为抄袭的文章
    bn_ratio=bn_model.verify_atricle(total_predcitons)
    print('朴素贝叶斯结束')
    print('-'*40)
    print('伯努利模型开始')
    # 伯努利模型
    bnb_model=MyKnn_Verify_Copy_News(model_type='bnb')
    training_score, testing_score, total_predcitons = bnb_model.main()
    original_labels = bnb_model.get_original_labels()
    bnb_ratio=bnb_model.verify_atricle(total_predcitons)
    print('伯努利模型结束')

    if bn_ratio>bnb_ratio:
        print('同样本下朴素贝叶斯模型更好')
    elif bn_ratio<bnb_ratio:
        print('同样本下伯努利模型更好')
    else:
        print('同样本下两个模型效果一致')







