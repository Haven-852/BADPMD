from gensim import corpora, models
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ldamodel(num_topics, pwd):
    df = pd.read_excel(pwd)
    train = [text.split() for text in df['raw_data'].dropna()]
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    corpora.MmCorpus.serialize('corpus.mm', corpus)
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, random_state=1,
                          num_topics=num_topics)

    topic_list = lda.print_topics(num_topics, 10)

    return lda, dictionary

def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    print('the info of this ldamodel: \n')
    print('num of topics: %s' % num_topics)
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = []
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = []
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0
        doc = testset[i]
        doc_word_num = 0
        for word_id, num in dict(doc).items():
            prob_word = 0.0
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic * prob_topic_word
            prob_doc += math.log(prob_word)  # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum / testset_word_num)  # perplexity = exp(-sum(p(d)/sum(Nd))
    print("模型困惑度为 : %s" % prep)
    return prep

def graph_draw(topic, perplexity):
    x = topic
    y = perplexity
    plt.plot(x, y, color="red", linewidth=2)
    plt.xticks(np.arange(min(x), max(x) + 1, 1))
    plt.xlabel("Number of Topic")
    plt.ylabel("perplexity")
    plt.savefig("困惑度主题数量")
    plt.show()

if __name__ == '__main__':
    pwd = '..//data//data_DPP-review.xlsx'
    for i in range(20,21,1):
        print("抽样为"+str(i)+"时的perplexity")
        a=range(1,11,1) # 主题个数
        p=[]
        for num_topics in a:
            lda, dictionary = ldamodel(num_topics, pwd)
            corpus = corpora.MmCorpus('corpus.mm')
            testset = []
            for c in range(int(corpus.num_docs/i)):
                testset.append(corpus[c*i])
            prep = perplexity(lda, testset, dictionary, len(dictionary.keys()), num_topics)
            p.append(prep)
        graph_draw(a,p)

    import matplotlib.pyplot as plt
    import numpy as np

    # Simulate the perplexity data based on the provided image and its apparent trend
    topic = np.array(range(1, 11))  # Number of topics from 1 to 10
    perplexity = np.linspace(150, 90, num=10)  # Dummy perplexity values from the image

    # Calculate the slopes between each pair of consecutive points
    slopes = np.diff(perplexity) / np.diff(topic)

    # Plot the original perplexity curve
    plt.plot(topic, perplexity, color="red", linewidth=2)
    plt.xticks(np.arange(min(topic), max(topic) + 1, 1))
    plt.xlabel("Number of Topic")
    plt.ylabel("Perplexity")

    # Adding slope values to the plot for each segment between points
    for i, slope in enumerate(slopes, 1):
        plt.annotate(f"Slope: {slope:.2f}", (topic[i], perplexity[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center')

    plt.show()
