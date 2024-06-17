from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from langdetect import detect_langs
from symspellpy import SymSpell, Verbosity
import pkg_resources
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#SymSpell 的主要功能是根据给定的词典和编辑距离阈值，为输入的单词找到可能的正确拼写。
sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


# lowercase + base filter
# some basic normalization
def f_base(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # 确保输入是字符串
    if not isinstance(s, str):
        s = str(s)

    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: "&gt", "&lt"
    s = re.sub(r'&gt|&lt', ' ', s)
    # normalization 4: letter repetition (if more than 2)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # normalization 5: non-word repetition (if more than 1)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s)
    # normalization 6: string * as delimiter
    s = re.sub(r'\*|\W\*|\*\W', '. ', s)
    # normalization 7: stuff in parenthesis, assumed to be less informal
    # s = re.sub(r'\(.*?\)', '. ', s)
    # normalization 8: xxx[?!]. -- > xxx.
    s = re.sub(r'\W+?\.', '.', s)
    # normalization 9: [.?!] --> [.?!] xxx
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    # normalization 10: ' ing ', noise text
    s = re.sub(r' ing ', ' ', s)
    # normalization 11: noise text
    # s = re.sub()
    # normalization 12: phrase repetition
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)

    return s.strip()
# language detection
#这个函数的目的是检测输入字符串是否为英语，如果是则返回 True，否则返回 False。

def f_lan(s):
    """
    :param s: string to be processed
    :return: boolean (s is English)
    """
    # some reviews are actually english but biased toward french
    return detect_langs(s) in {'English'}

###############################
#### word level preprocess ####
###############################

# filtering out punctuations and numbers
#用于对单词列表进行预处理，以去除其中的标点符号和数字，保留纯字母单词
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    return [word for word in w_list if word.isalpha()]

#用于从输入的单词列表中选择名词和形容词，并返回只包含这些词性的单词列表
# selecting nouns
def f_noun(w_list, rare_words_file=None):
    """
    选择名词
    :param w_list: 待处理的单词列表
    :param rare_words_file: 包含生僻单词的文件路径
    :return: 只包含名词或形容词的单词列表
    """
    # 使用 NLTK 库中的 pos_tag 函数对单词列表进行词性标注，标注结果存储在 pos 列表中
    # 使用列表推导式，遍历 pos 列表中的每个元素，元素为 (word, pos) 形式的元组，
    # 如果词性标注的前两个字符（即词性的缩写）在 ['NN','JJ'] 中（名词或形容词），则保留该单词

    rare_words = []

    if rare_words_file:
        with open(rare_words_file, 'r') as file:
            for line in file:
                rare_words.append(line.strip())

    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] in ['NN','JJ'] or word.lower() in rare_words]



# typo correction
#这段代码主要用于对单词列表进行拼写纠错，并返回经过纠错后的单词列表。
def f_typo(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with typo fixed by symspell. words with no match up will be dropped
    """
    w_list_fixed = []
    for word in w_list:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
        if suggestions:
            w_list_fixed.append(suggestions[0].term)
        else:
            pass
            # do word segmentation, deprecated for inefficiency
            # w_seg = sym_spell.word_segmentation(phrase=word)
            # w_list_fixed.extend(w_seg.corrected_string.split())
    return w_list_fixed

#一个用于词干化单词列表，另一个用于过滤停用词。
# stemming if doing word-wise
p_stemmer = PorterStemmer()


def f_stem(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    return [p_stemmer.stem(word) for word in w_list]


# filtering out stop words
# create English stop words list
nltk.download('stopwords')
en_stop = stopwords.words('english')
#en_stop.append('')

#用于过滤掉输入的单词列表中的停用词，并返回过滤后的单词列表
def f_stopw(w_list):
    """
    filtering out stop words
    """
    return [word for word in w_list if word not in en_stop]


#用于从原始的评论文本中获取经过句子级别预处理后的数据，并返回处理后的结果
def preprocess_sent(rw_list):
    """
    Get sentence level preprocessed data from a list of raw review texts
    :param rw_list: list of reviews to be processed
    :return: list of sentence level pre-processed reviews
    """
    """
        从原始评论文本的列表中获取句子级别预处理后的数据
        :param rw_list: 待处理的评论列表
        :return: 经过句子级别预处理后的评论列表
    """
    processed_reviews = []
    for rw in rw_list:
        # 使用 f_base 函数对原始评论文本进行基本的预处理（如大小写转换、标点符号处理等）
        s = f_base(rw)
        # 使用 f_lan 函数检测评论文本是否为英语，如果不是英语则返回 None
        #if not f_lan(s):
            #continue  # 如果不是英语评论，则跳过当前评论
        processed_reviews.append(s)
    return processed_reviews

#用于从预处理过的句子中获取经过词级别预处理后的数据，并返回处理后的结果
def preprocess_word(s_list):
    """
    Get word level preprocessed data from a list of preprocessed sentences
    including: remove punctuation, select noun, fix typo, stem, stop_words
    :param s_list: list of sentences to be processed
    :return: list of word level pre-processed reviews
    """
    """
    从预处理过的句子的列表中获取经过词级别预处理后的数据，包括：去除标点符号、选择名词、修正拼写错误、词干化、停用词过滤
    :param s_list: 待处理的句子列表
    :return: 经过词级别预处理后的句子列表
    """
    processed_sentences = []
    for s in s_list:
        if not s:    # 如果输入的句子为空，则跳过当前句子
            continue
        w_list = word_tokenize(s)  # 使用 word_tokenize 函数将句子分词成单词列表
        # 使用预处理函数对单词列表进行词级别的预处理
        w_list = f_punct(w_list)  # 去除标点符号
        w_list = f_noun(w_list)   # 选择名词
        w_list = f_typo(w_list)   # 修正拼写错误
        #w_list = f_stem(w_list)
        w_list = f_stopw(w_list)  # 停用词过滤
        processed_sentences.append(w_list)

    return processed_sentences   # 返回经过词级别预处理后的句子列表

def f_lan_1(s):
    """
    Detect if the input string is in English.
    :param s: string to be processed
    :return: boolean, True if the string is in English, False otherwise
    """
    # Detect the language of the string and return True if English is detected.
    # The detect_langs function returns a list of Language objects with confidence scores.
    # We convert the list to string representations and check if 'en' (English) is present.
    return any('en' in str(lang) for lang in detect_langs(s))
def preprocess_sent_1(rw):
    """
    Perform base preprocessing of the text at the sentence level.
    :param rw: Raw text string to preprocess.
    :return: Preprocessed text string.
    """
    s = f_base(rw)
    if f_lan_1(s):
        return s
    else:
        return None

# Now, let's redefine preprocess_word for a single preprocessed text string:
def preprocess_word_1(s):
    """
    Perform word level preprocessing given a preprocessed text string.
    :param s: Preprocessed text string.
    :return: List of words after further preprocessing.
    """
    if not s:
        return []
    w_list = word_tokenize(s)
    w_list = f_punct(w_list)
    w_list = f_noun(w_list)
    w_list = f_typo(w_list)
    # w_list = f_stem(w_list)
    w_list = f_stopw(w_list)
    return w_list