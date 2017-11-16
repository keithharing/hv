from gensim.models.keyedvectors import KeyedVectors
import scipy.stats as st
import argparse


# Word Similarity test
def word_sim_test(filename, word_vectors):
    delim = ','
    actual_sim_list, pred_sim_list = [], []
    missed = 0

    with open(filename, 'r',encoding='utf-8') as pairs:
        for pair in pairs:
            w1, w2, actual_sim = pair.strip().split(delim)

            try:
                pred = word_vectors.similarity(w1, w2)
                actual_sim_list.append(float(actual_sim))
                pred_sim_list.append(pred)

            except KeyError:
                missed += 1

    spearman, _ = st.spearmanr(actual_sim_list, pred_sim_list)
    pearson, _ = st.pearsonr(actual_sim_list, pred_sim_list)

    return spearman, pearson, missed


# Word Analogy Test
def analogy_test(filename, word_vectors):
    correct = 0
    total = 0
    with open(filename, 'r',encoding='utf-8') as lines:
        for line in lines:
            if line.startswith("#") or len(line) <= 1:
                continue

            words = line.strip().split(" ")

            total += 1
            try:
                similar_words = word_vectors.most_similar(positive=[words[1], words[2]], negative=[words[0]])
#                if similar_words[0][0].startswith(words[3]):
                if similar_words[0][0] == words[3]:
                    correct += 1

            except KeyError:
                pass

    return correct, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vector_file", type=str, help="word vector input")
    args = parser.parse_args()

    word_vectors = KeyedVectors.load_word2vec_format(args.vector_file, binary=False)

    spearman, pearson, missed = word_sim_test("./test_dataset/kor_ws353.csv", word_vectors)
    print("===== Word Similarity Test ====")
    print("Missed :", missed)
    print("Spearman :", spearman)
    print("Pearson :", pearson)
    print()

    print("====== Word Analogy Test ======")
    print("Semantic : ")
    correct, total = analogy_test("./test_dataset/kor_analogy_semantic.txt", word_vectors)
    print(str(correct) + "/" + str(total) + " = " + str(correct/total))
    print("Syntactic : ")
    correct, total = analogy_test("./test_dataset/kor_analogy_syntactic.txt", word_vectors)
    print(str(correct) + "/" + str(total) + " = " + str(correct/total))

