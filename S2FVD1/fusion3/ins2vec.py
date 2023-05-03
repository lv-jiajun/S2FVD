import os
import logging

from gensim import models
from gensim.models import Word2Vec, KeyedVectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
                    if not line.startswith(">>>"):  # is not summary line
                        yield line.split()


def trainWord2Vec(corpus_path, dic_file_path, suffix, save_whole_model=True):
    """
    obtain a phaseII dictionary with skip-gram model
    :param corpus_path:
    :param dic_file_path:
    :param suffix:
    :param save_whole_model: default True, save the whole model. otherwise just save the standalone keyed vectors
    :return:
    """
    texts = MyCorpus(corpus_path, suffix)
    model = Word2Vec(texts, size=100, window=5, min_count=5, workers=4, sg=1)
    if save_whole_model:
        model.save(dic_file_path)
    else:
        model.wv.save(dic_file_path)


def show_vectors(model_path, whole=True):
    if whole:
        model = models.Word2Vec.load(model_path)
        word_vectors = model.wv
    else:
        word_vectors = KeyedVectors.load(model_path, mmap='r')
    print(word_vectors.vector_size)
    print(len(word_vectors.vocab))
    word_vectors.init_sims()
    for word in word_vectors.index2word:
        print(word)
        print(word_vectors.wv.word_vec(word, use_norm=True))
        print(word_vectors.wv.word_vec(word, use_norm=False))


if __name__ == '__main__':
    # corpus_path = 'F:/tmp/compilerprovenance'
    corpus_path = 'C:/Users/lvjiajun/Desktop/file/code'
    # dic_file_path = 'F:/tmp/compilerprovenance/ins2vec_coarse.dic'
    dic_file_path = 'C:/Users/lvjiajun/Desktop/file/code/ins2vec_coarse.dic'
    suffix = 'lvtoken1.csv'
    trainWord2Vec(corpus_path, dic_file_path, suffix)
    # show_vectors(dic_file_path)

    # dic_file_path = 'F:/tmp/compilerprovenance/ins2vec_fine.dic'
    # suffix = 'BB#fine.csv'
    # trainWord2Vec(corpus_path, dic_file_path, suffix)
    # show_vectors(dic_file_path)


