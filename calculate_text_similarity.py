import json
import numpy as np
from stages.tokenization import annotate_processes, filter_punctuation_marks, build_matrizes, build_csv_matrix
from stages.preprocess import get_X_y
from stages.use import get_compressions

def read_process_description_data(data_source):
    data_string = open(data_source, 'rb').read().decode('utf-8')
    json_objects = json.loads(data_string)
    print('Number of sentences:', len(json_objects))
    return json_objects

def calculate_bagofword_similarity(bag1, bag2):
    # TODO add meaningful implementation
    union = set()
    for word in bag1:
        union.add(word)
    for word in bag2:
        union.add(word)
    similarity_matrix = populate_similarity_matrix(union)
    sc_1 = calculate_soft_cardinality(bag1, similarity_matrix, 1)
    sc_2 = calculate_soft_cardinality(bag2, similarity_matrix, 1)
    sc_union = calculate_soft_cardinality(union, similarity_matrix, 1)
    sc_intersect = sc_1 + sc_2 - sc_union
    return (calculate_jaccard(sc_intersect, sc_union), calculate_dice(sc_intersect, sc_1, sc_2), calculate_cosine(sc_intersect, sc_1, sc_2))

def calculate_jaccard(len_intersect, len_union):
    return len_intersect/len_union

def calculate_dice(len_intersect, len_a, len_b):
    return (2*len_intersect)/(len_a + len_b)

def calculate_cosine(len_intersect, len_a, len_b):
    return len_intersect/((len_a * len_b)**0.5)

def calculate_soft_cardinality(set_of_words, similarity_matrix, p):
    result = 0
    for word in set_of_words:
        sum = 0
        for ref in set_of_words:
            sum += similarity_matrix[word][ref]**p
        result += 1/sum
    return result

def populate_similarity_matrix(words):
    res = {}
    for word in words:
        for ref in words:
            if not (word in res):
                res[word] = {}
            res[word][ref] = calculate_syntactical_word_similarity(word, ref)
    return res


def calculate_syntactical_word_similarity(word_a, word_b):
    edit_distance = levenshtein(word_a, word_b)
    max_len = max([len(word_a), len(word_b)])
    return 1-(edit_distance/max_len)

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

if __name__ == '__main__':
    calculate_bagofword_similarity(np.array(['this', 'is', 'a', 'test']), np.array(['another', 'test']))
    # Tokenize corpus
    PD_DATA_SOURCES = ['process1.json', 'process2.json', 'process3.json']
    XY = []
    bows = []
    for data_source in PD_DATA_SOURCES:
        print('Reading ', data_source)
        data = read_process_description_data('ressources/process_descriptions/' + data_source)
        sentences, compressions, labels = annotate_processes(data)
        X = build_matrizes(sentences, compressions)
        header_row = ['word', 'pos', 'dependency_label', 'id', 'parent', 'EOS', 'kept']
        data = build_csv_matrix(header_row, X)
        data.to_csv('csv/process_descriptions/' + data_source + '.csv', sep=',', encoding='utf-8', na_rep='N/A')
        print('Totally saved:', len(X))
        XY.extend(X)
        for i in range(len(labels)):
            label_bows = []
            for j in range(len(labels[i])):
                labels[i][j] = filter_punctuation_marks(labels[i][j])
                bag_of_words = []
                for z in range(len(labels[i][j])-1):
                    bag_of_words.append(labels[i][j][z]['word'])
                label_bows.append(np.array(bag_of_words))
            bows.append(label_bows)

    # Prepcocess and predict on data
    use_syn_feat = True
    X, y = get_X_y(XY)

    predicted_compressions_with_synfeat = get_compressions(X, use_synfeat=True)
    # Transform data to bag of words
    for i in range(len(predicted_compressions_with_synfeat)):
        predicted_compressions_with_synfeat[i] = predicted_compressions_with_synfeat[i][:-1, 0]

    predicted_compressions_wo_synfeat = get_compressions(X, use_synfeat=False)
    # Transform data to bag of words
    for i in range(len(predicted_compressions_wo_synfeat)):
        predicted_compressions_wo_synfeat[i] = predicted_compressions_wo_synfeat[i][:-1, 0]

    # Transform data to bag of words
    for i in range(len(X)):
        X[i] = X[i][:-1, 0]

    scores_syn = []
    scores_no_syn = []
    scores_original = []
    header = ['label', 'original', 'compression_wo_syn', 'compression_with_syn', 'dice_original', 'dice_wo_syn', 'dice_with_syn', 'jaccard_original', 'jaccard_wo_syn', 'jaccard_with_syn', 'cosine_original', 'cosine_wo_syn', 'cosine_with_syn']
    res = []
    no_tokens_raw = 0
    no_tokens_labels = 0
    for i in range(len(bows)):
        for label in bows[i]:
            no_tokens_labels += len(label)
            no_tokens_raw += len(X[i])
            scores_syn.append(calculate_bagofword_similarity(label, predicted_compressions_with_synfeat[i]))
            scores_no_syn.append(calculate_bagofword_similarity(label, predicted_compressions_wo_synfeat[i]))
            scores_original.append(calculate_bagofword_similarity(label, X[i]))
            label_string = ' '.join(label)
            comp_string_no_syn = ' '.join(predicted_compressions_wo_synfeat[i])
            comp_string_syn = ' '.join(predicted_compressions_with_synfeat[i])
            ori_string = ' '.join(X[i])
            res.append([label_string, ori_string, comp_string_no_syn, comp_string_syn, scores_original[-1][1], scores_no_syn[-1][1], scores_syn[-1][1], scores_original[-1][0], scores_no_syn[-1][0], scores_syn[-1][0], scores_original[-1][2], scores_no_syn[-1][2], scores_syn[-1][2]])

    print(no_tokens_raw)
    print(no_tokens_labels)
    print('Ratio:', no_tokens_labels/no_tokens_raw)
    res = np.array(res)
    print(res)
    scores = res[:, 4:]
    scores.astype(np.float64)

    print('Averace dice original:', np.mean(scores[:, 0].astype(np.float64)))
    print('Averace dice without syn:', np.mean(scores[:, 1].astype(np.float64)))
    print('Averace dice with syn:', np.mean(scores[:, 2].astype(np.float64)))
    print('Averace jaccard original:', np.mean(scores[:, 3].astype(np.float64)))
    print('Averace jaccard without syn:', np.mean(scores[:, 4].astype(np.float64)))
    print('Averace jaccard with syn:', np.mean(scores[:, 5].astype(np.float64)))
    print('Averace cosine original:', np.mean(scores[:, 6].astype(np.float64)))
    print('Averace cosine without syn:', np.mean(scores[:, 7].astype(np.float64)))
    print('Averace cosine with syn:', np.mean(scores[:, 8].astype(np.float64)))

