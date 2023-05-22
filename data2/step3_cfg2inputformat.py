import os
from gensim.models import Word2Vec

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';',
    '{', '}'
}

def tokenize(line):
    tmp, w = [], []
    i = 0
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 3])
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 2])
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    # Filter out irrelevant strings
    res = list(filter(lambda c: c != '', tmp))
    return list(filter(lambda c: c != ' ', res))


class MyCorpus(object):
    """ Data Preparation \n
    gensim’s word2vec expects a sequence of sentences as its input. Each sentence is a list of words (utf8 strings).

    Gensim only requires that the input must provide sentences sequentially, when iterated over. No need to keep everything
    in RAM: we can provide one sentence, process it, forget it, load another sentence...

    Say we want to further preprocess the words from the files — convert to unicode, lowercase, remove numbers, extract
    named entities… All of this can be done inside the MySentences iterator and word2vec doesn’t need to know. All that is
    required is that the input yields one sentence (list of utf8 words) after another.
    """

    def __init__(self, dirname, suffix):
        self.dirname = dirname
        self.suffix = suffix

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if fname.endswith(self.suffix):  # filter irrelevant files
                for line in open(os.path.join(self.dirname, fname)):
                    # print(line)
                    if not line.startswith(">>>"):  # is not summary line
                        yield line.split()


def trainWord2Vec(corpus_path, dic_file_path, suffix):
    """
    obtain a phaseII dictionary with skip-gram model
    :param corpus_path:
    :param dic_file_path:
    :param suffix:
    :param save_whole_model: default True, save the whole model. otherwise just save the standalone keyed vectors
    :return:
    """
    texts = MyCorpus(corpus_path, suffix)
    print("training word2vec models........")
    model = Word2Vec(texts, size=100, window=5, min_count=5, workers=96, sg=1)
    # model = models.FastText(texts, size=100, window=5, min_count=5,  sg=0, min_n=2, max_n=3)

    # if save_whole_model:
    model.save(dic_file_path)
    model.wv.save_word2vec_format('word2vec.vector')


def merge_all_cfgs():
    filejia = "/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/cfgs/pdgs"
    f_new = open('/data/bhtian2/win_linux_mapping/three-fusion/data2/merge.csv', 'w', encoding='utf-8')
    files = os.listdir(filejia)
    # files.sort(key=lambda x: int(x[:-6]))
    files.sort(key=lambda x: int(x.split("_")[0]))
    index = 1
    for f in files:
            if f.endswith(".dot"):
                flag = 1
                map_a = dict()
                k = f.split('_')
                label = f.split(".")[0].split("_")[1]
                sum = k[0]
                i = 0
                with open(filejia+'/'+f, "r+", encoding="utf8") as file:
                    index = index + 1
                    f_new.write('\n')
                    f_new.write('*'*26)
                    for line in file:
                        kv = line.strip().split(' ')
                        if kv[0] == "digraph":
                            # func = kv[1].split("\"")[1]
                            func = kv[1]
                            flag = 0
                            f_new.write('\n')
                        elif flag == 0:
                            str1 = "[label = "
                            if str1 in line:
                                kv1 = line.strip().split(' [label = ')
                                map_a[kv1[0]] = i
                                i = i+1
                                f_new.write(kv1[1][:-1])
                                f_new.write('\n')
                            else:
                                f_new.write(">>>func&"+str(func))
                                f_new.write('\n')
                                f_new.write(">>>cfg&"+str(sum))
                                flag = 2
                                str2 = '->'
                                if str2 in line:
                                    kv2 = line.strip().split(' -> ')
                                    n1 = map_a[kv2[0]]
                                    n2 = map_a[kv2[1]]
                                    f_new.write(" ")
                                    f_new.write(str(n1) + '->' + str(n2))
                        elif flag == 2:
                            str2 = '->'
                            if str2 in line:
                                kv2 = line.strip().split(' -> ')
                                n1 = map_a[kv2[0]]
                                n2 = map_a[kv2[1]]
                                f_new.write(" ")
                                f_new.write(str(n1)+'->'+str(n2))
                    f_new.write('\n>>>func&label&#' + label)
    print("index", str(index))


def tokenize_cfg_nodes():
    sum = 0
    filename = '/data/bhtian2/win_linux_mapping/three-fusion/data2/merge.csv'
    file_new = open('/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/cfgs/new1hash_1CFG#coarse.csv', 'w', encoding='utf-8')
    with open(filename, "r+", encoding="utf8") as file:
        for line in file:
            if line.startswith(">>>cfg&"):
                sum = sum + 1

            stripped = line.strip()
            if not stripped:  # 判断是否是空，如果是则continue
                continue
            elif stripped.startswith("**************************"):
                continue
            elif stripped.startswith(">>>func&"):
                file_new.write(line)
            elif stripped.startswith(">>>cfg&"):
                file_new.write(line)
            else:
                line = line[1:-2]
                line1 = tokenize(line)
                file_new.write(str(line1))
                file_new.write('\n')

    print(sum)
    file.close()

    corpus_path = '/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/cfgs'
    dic_file_path = '/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/cfgs/ins2vec_coarse.dic'
    suffix = 'new1hash_1CFG#coarse.csv'
    trainWord2Vec(corpus_path, dic_file_path, suffix)


if __name__ == '__main__':
    merge_all_cfgs()
    tokenize_cfg_nodes()