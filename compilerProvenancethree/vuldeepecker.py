"""
Interface to VulDeePecker project
"""
import sys
import os
import pandas
from TextCp import TextCnn
from clean_gadget import clean_gadget
from vectorize_gadget import GadgetVectorizer
import os
from gensim.models import Word2Vec, FastText

"""
将多个程序片组合成一个code gadget，然后对每个code gadget标注，有漏洞的标为“1”，没有漏洞的标为“0”
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
解析小工具文件以查找单个小工具
将每个小工具生成为字符串列表，其中每个元素都是代码行
必须忽略每个小工具的第一行，该行以整数+空格开头
在每个代码小工具的末尾是二进制值
     这表明该小工具中是否存在漏洞
"""
'''
def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        gadget = []
        flag = 0
        count = 0
        sum = 0
        for line in file:
            stripped = line.strip()   #去除首尾的空格
            if not stripped:          #判断是否是空，如果是则continue
                continue
            if "-" * 26 in line and gadget:
                if flag == 6:
                    sum = sum+1
                    yield clean_gadget(
                        gadget), gadget_val1, gadget_val2, gadget_val3, gadget_val4, gadget_val5
                    flag = 0
                else:
                    flag = flag+1
                gadget = []
            elif str("CWE-119") in stripped:
                if str("CWE-119  True") in stripped:
                    flag = 6
                    count = count+1
                gadget_val1 = stripped
            elif str("CWE-120") in stripped:
                if str("CWE-120  True") in stripped:
                    flag = 6
                    count = count + 1
                gadget_val2 = stripped
            elif str("CWE-469") in stripped:
                if str("CWE-469  True") in stripped:
                    flag = 6
                    count = count + 1
                gadget_val3 = stripped
            elif str("CWE-476") in stripped:
                if str("CWE-476  True") in stripped:
                    flag = 6
                    count = count + 1
                gadget_val4 = stripped
            elif str("CWE-other") in stripped:
                if str("CWE-other  True") in stripped:
                    flag = 6
                    count = count + 1
                gadget_val5 = stripped
            else:
                gadget.append(stripped)
        print("有漏洞的函数个数：", count)
        print("总函数个数：", sum)
'''
'''
def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        gadget = []
        flag = 0
        count = 0
        sum = 0
        for line in file:
            stripped = line.strip()   #去除首尾的空格
            if not stripped:          #判断是否是空，如果是则continue
                continue
           # if "-" * 26 in stripped and gadget:
            if line.startswith('--------------------------') and line.endswith('--------------------------\n') and gadget:
            #if stripped.startswith("-" * 26) and gadget:
                sum = sum+1
                yield clean_gadget(
                        gadget), gadget_val1, gadget_val2, gadget_val3, gadget_val4, gadget_val5
                gadget = []
            elif str("CWE-119") in stripped:
                if str("CWE-119 True") in stripped:
                    count = count+1
                gadget_val1 = stripped
            elif str("CWE-120") in stripped:
                if str("CWE-120 True") in stripped:
                    count = count + 1
                gadget_val2 = stripped
            elif str("CWE-469") in stripped:
                if str("CWE-469 True") in stripped:
                    count = count + 1
                gadget_val3 = stripped
            elif str("CWE-476") in stripped:
                if str("CWE-476 True") in stripped:
                    count = count + 1
                gadget_val4 = stripped
            elif str("CWE-other") in stripped:
                if str("CWE-other True") in stripped:
                    count = count + 1
                gadget_val5 = stripped
            else:
                gadget.append(stripped)
        print("总函数个数：", sum)
'''


def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        gadget = []
        flag = 0
        count = 0
        sum = 0
        for line in file:

            stripped = line.strip()  # 去除首尾的空格
            if not stripped:  # 判断是否是空，如果是则continue
                continue
            gadget = stripped[:-2]
            gadget_val1 = stripped[-1]
            gadget_val2 = stripped[-2]
            if (gadget_val1.startswith("0")) and gadget_val2.startswith("#"):
                gadget_val1 = "0"
            elif (gadget_val1.startswith("1")) and gadget_val2.startswith("#"):
                gadget_val1 = "1"
            elif (gadget_val1.startswith("2")) and gadget_val2.startswith("#"):
                gadget_val1 = "2"
            elif (gadget_val1.startswith("3")) and gadget_val2.startswith("#"):
                gadget_val1 = "3"
            elif (gadget_val1.startswith("4")) and gadget_val2.startswith("#"):
                gadget_val1 = "4"
            elif (gadget_val1.startswith("5")) and gadget_val2.startswith("#"):
                gadget_val1 = "5"
            elif (gadget_val1.startswith("6")) and gadget_val2.startswith("#"):
                gadget_val1 = "6"
            elif (gadget_val1.startswith("7")) and gadget_val2.startswith("#"):
                gadget_val1 = "7"
            elif (gadget_val1.startswith("8")) and gadget_val2.startswith("#"):
                gadget_val1 = "8"
            elif (gadget_val1.startswith("9")) and gadget_val2.startswith("#"):
                gadget_val1 = "9"
            elif (gadget_val1.startswith("0")) and gadget_val2.startswith("1"):
                gadget_val1 = "10"
            elif (gadget_val1.startswith("1")) and gadget_val2.startswith("1"):
                gadget_val1 = "11"
            elif (gadget_val1.startswith("2")) and gadget_val2.startswith("1"):
                gadget_val1 = "12"
            elif (gadget_val1.startswith("3")) and gadget_val2.startswith("1"):
                gadget_val1 = "13"
            elif (gadget_val1.startswith("4")) and gadget_val2.startswith("1"):
                gadget_val1 = "14"
            elif (gadget_val1.startswith("5")) and gadget_val2.startswith("1"):
                gadget_val1 = "15"
            elif (gadget_val1.startswith("6")) and gadget_val2.startswith("1"):
                gadget_val1 = "16"
            elif (gadget_val1.startswith("7")) and gadget_val2.startswith("1"):
                gadget_val1 = "17"
            elif (gadget_val1.startswith("8")) and gadget_val2.startswith("1"):
                gadget_val1 = "18"
            elif (gadget_val1.startswith("9")) and gadget_val2.startswith("1"):
                gadget_val1 = "19"
            elif (gadget_val1.startswith("0")) and gadget_val2.startswith("2"):
                gadget_val1 = "20"
            elif (gadget_val1.startswith("1")) and gadget_val2.startswith("2"):
                gadget_val1 = "21"
            elif (gadget_val1.startswith("2")) and gadget_val2.startswith("2"):
                gadget_val1 = "22"
            elif (gadget_val1.startswith("3")) and gadget_val2.startswith("2"):
                gadget_val1 = "23"
            elif (gadget_val1.startswith("4")) and gadget_val2.startswith("2"):
                gadget_val1 = "24"
            elif (gadget_val1.startswith("5")) and gadget_val2.startswith("2"):
                gadget_val1 = "25"
            yield gadget, gadget_val1
'''
def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        gadget = []
        flag = 0
        count = 0
        sum = 0
        for line in file:

            stripped = line.strip()   #去除首尾的空格
            if not stripped:          #判断是否是空，如果是则continue
                continue
            gadget = stripped[:-2]
            gadget_val1 = stripped[-1]
            yield gadget, gadget_val1
'''
"""
Uses gadget file parser to get gadgets and vulnerability indicators
Assuming all gadgets can fit in memory, build list of gadget dictionaries
    Dictionary contains gadgets and vulnerability indicator
    Add each gadget to GadgetVectorizer
Train GadgetVectorizer model, prepare for vectorization
Loop again through list of gadgets
    Vectorize each gadget and put vector into new list
Convert list of dictionaries to dataframe when all gadgets are processed
使用小工具文件解析器获取小工具和漏洞指示器
假设所有小工具都可以容纳在内存中，则建立小工具字典列表  词典包含小工具和漏洞指示器
将每个小工具添加到GadgetVectorizer
训练GadgetVectorizer模型，为向量化做准备
再次循环浏览小工具列表
     向量化每个小工具并将向量放入新列表
处理完所有小工具后，将字典列表转换为数据框
"""
def get_vectors_df(filename, vector_length=100):
    gadgets = []
    count = 0
    vectorizer = GadgetVectorizer(vector_length)
    for gadget, val1 in parse_file(filename):
        count += 1
        #print(gadget)
        print("Collecting gadgets...", count, end="\r")
        gadget = vectorizer.add_gadget(gadget) #判断是前向切片还是后向切片
        row = {"gadget" : gadget, "val1" : val1}
        gadgets.append(row)
    '''
    for gadget, val1, val2, val3, val4, val5 in parse_file(filename):
        count += 1
        #print(gadget)
        print("Collecting gadgets...", count, end="\r")
        gadget1 = vectorizer.add_gadget(gadget) #判断是前向切片还是后向切片
        row = {"gadget" : gadget1, "val1" : val1,"val2" : val2,"val3" : val3,"val4" : val4,"val5" : val5}
        gadgets.append(row)
    '''
    #txt = open("./lvtoken1.txt", "w").write(str(gadgets))
    df = pandas.DataFrame(gadgets)
    outputpath = 'D:/JiajunLv/code3/lvtoken2.csv'
    df.to_csv(outputpath, sep=',', index=True, header=True)
    return df


class MyCorpus(object):
    """ Data Preparation \n
    gensim’s word2vec expects a sequence of sentences as its input. Each sentence is a list of words (utf8 strings).

    Gensim only requires that the input must provide sentences sequentially, when iterated over. No need to keep everything
    in RAM: we can provide one sentence, process it, forget it, load another sentence...

    Say we want to further preprocess the words from the files — convert to unicode, lowercase, remove numbers, extract
    named entities… All of this can be done inside the MySentences iterator and word2vec doesn’t need to know. All that is
    required is that the input yields one sentence (list of utf8 words) after another.
    """

    def __init__(self, df, suffix):
        self.df = df
        self.suffix = suffix

    def __iter__(self):
        '''
        for fname in os.listdir(self.dirname):
            if fname.endswith(self.suffix):  # filter irrelevant files
                for line in open(os.path.join(self.dirname, fname)):
                    if not line.startswith(">>>"):  # is not summary line
                        yield line.split()
        '''
        for i in range(0, len(self.df)):
            for line in self.df.iloc[i]['gadget']:
                 yield line.split(' ')

def trainWord2Vec(df, dic_file_path, suffix, save_whole_model=True):
    """
    obtain a phaseII dictionary with skip-gram model
    :param corpus_path:
    :param dic_file_path:
    :param suffix:
    :param save_whole_model: default True, save the whole model. otherwise just save the standalone keyed vectors
    :return:
    """
    texts = MyCorpus(df, suffix)
    model = Word2Vec(texts, size=100, window=5, min_count=3, workers=4, sg=1)
    #model = FastText(texts, size=100, window=5, min_count=5,  sg=1)
    if save_whole_model:
        model.save(dic_file_path)
        model.wv.save_word2vec_format('D:/JiajunLv/code2//word2vec.vector')

    else:
        model.wv.save_word2vec_format(dic_file_path, binary=False)

def Part1(filename,vec):
    #filename = 'E:/model/JiajunLv/code1/sample.txt'
    parse_file(filename)
    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_gadget_vectors.pkl"   #这两行代码进行文件后缀名分割，如将cwe119_cgd.txt变为cwe119_cgd_gadget_vectors.pkl
    vector_length = 50
   # if os.path.exists(vector_filename):
       # df = pandas.read_pickle(vector_filename)   #pandas库pd.read_pickle操作读取pickle数据与.to_pickle()永久储存数据
    #else:
    df = get_vectors_df(filename, vector_length)
    dic_file_path = 'D:/JiajunLv/code2/ins2vec_coarse.dic'
    #dic_file_path = 'E:/model/JiajunLv/code2/ins2vec_coarse.dic'
    suffix = 'lvtoken2.csv'
    trainWord2Vec(df, dic_file_path, suffix)
    texts,embeddings_matrix,fig_prefix = TextCnn(df,vec)
    return texts,embeddings_matrix,fig_prefix

if __name__ == "__main__":
    Part1()