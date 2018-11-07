import subprocess
import requests
import time
import progressbar
import pandas as pd

# Run the sentence parsing REST API
subprocess.Popen(['docker', 'run', '-p', '4000:80', 'sentence_parser_rest'])
# os.system('docker run -p 4000:80 sentence_parser_rest')

def annotate_data(data):
    sentences = []
    compressions = []
    start = time.time()
    print("Started")

    LEN_DATA = len(data)
    p_bar = progressbar.ProgressBar(max_value=(LEN_DATA))

    for i in range(len(data)):
        sentences.append(get_annotation(data[i]['sentence']))
        if data[i]['compression']:
            compressions.append(get_annotation(data[i]['compression']))
        else:
            compressions.append([])
        p_bar.update(i + 1)

    end = time.time()
    print('Elapsed time:\t {}:{}'.format(int((end - start) / 60), int((end - start) % 60)))
    return (sentences, compressions)

def annotate_processes(process_data):
    sentences = []
    compressions = []
    labels = []
    start = time.time()
    print("Started")

    LEN_DATA = len(process_data)
    p_bar = progressbar.ProgressBar(max_value=(LEN_DATA))

    for i in range(len(process_data)):
        sentences.append(get_annotation(process_data[i]['sentence']))
        current_labels = []
        for label in process_data[i]['related_labels']:
            current_labels.append(get_annotation(label))
        labels.append(current_labels)
        if process_data[i]['compression']:
            compressions.append(get_annotation(process_data[i]['compression']))
        else:
            compressions.append([])
        p_bar.update(i + 1)

    end = time.time()
    print('Elapsed time:\t {}:{}'.format(int((end - start) / 60), int((end - start) % 60)))
    return (sentences, compressions, labels)


def get_annotation(sentence):
    sen_payload = {'sentence': sentence}
    req = requests.post('http://localhost:4000/parse', json=sen_payload)
    sen_tokens = req.json()
    return sen_tokens


def is_punctuation_mark(word):
    return (word['dependency_label'] == 'punct') and (word['pos'] != 'HYPH') and (word['word'] != '<EOS>')

def filter_punctuation_marks(sequence):
    result = []
    sequence.append({'word': '<EOS>', 'pos': '.', 'dependency_label': 'punct',
                   'id': sequence[len(sequence) - 1]['id'] if len(sequence) > 0 else 0,
                   'parent': sequence[len(sequence) - 1]['parent'] if len(sequence) > 0 else -1})
    for tmp in sequence:
        if is_punctuation_mark(tmp):
            continue
        else:
            result.append(tmp)
    return result

def generate_labels(source, target):
    source.append({'word': '<EOS>', 'pos': '.', 'dependency_label': 'punct', 'id': source[len(source) - 1]['id'],
                   'parent': source[len(source) - 1]['parent']})
    target.append({'word': '<EOS>', 'pos': '.', 'dependency_label': 'punct', 'id': target[len(target) - 1]['id'] if len(target) > 0 else 0,
                   'parent': target[len(target) - 1]['parent'] if len(target)>0 else -1})
    result = []
    current = target[0]
    while is_punctuation_mark(current) and len(target) > 0:
        target.pop(0)
        current = target[0]
    for i in range(len(source)):
        # Check if current word's dependency label is 'punct' and pos label is not 'HYPH'

        if is_punctuation_mark(source[i]):
            continue

        # Check if word comes next in the compressed sentence
        if source[i]['word'] == target[0]['word']:
            result.append([source[i]['word'], source[i]['pos'], source[i]['dependency_label'], source[i]['id'],
                           source[i]['parent'], 1])
            target.pop(0)
            if len(target) > 0:
                current = target[0]
                # Check if current word's dependency label in compression is 'punct' and pos label is not 'HYPH
                while is_punctuation_mark(current) and len(target) > 0:
                    target.pop(0)
                    current = target[0]
        else:
            result.append([source[i]['word'], source[i]['pos'], source[i]['dependency_label'], source[i]['id'],
                           source[i]['parent'], 0])
    try:
        assert (len(target) == 0), 'Not all words processed: ' + str(target)
    except AssertionError:
        try:
            print('Not all words processed:', str(target))
        except:
            print('Not all words processed.')
            print('Compression not decodable.')
            return None
        return None
    except UnicodeEncodeError:
        print('Unicode encode error')
        return None
    return result


def build_matrizes(sentences, compressions):
    dropped = 0
    res = []
    for sentence, compression in zip(sentences, compressions):
        sen_matrix = generate_labels(sentence, compression)
        if sen_matrix:
            res.append(sen_matrix)
        else:
            dropped += 1
    print('Dropped {} sentences'.format(dropped))
    return res


def build_csv_matrix(header_row, matrizes):
    data = []
    for sentence in matrizes:
        for i in range(len(sentence)):
            if i == (len(sentence) - 1):
                eos = 1
            else:
                eos = 0
            data.append(
                [sentence[i][0], sentence[i][1], sentence[i][2], sentence[i][3], sentence[i][4], eos, sentence[i][5]])
    indexes = list(range(len(data)))
    dataframe = pd.DataFrame(data=data, index=indexes, columns=header_row)
    return dataframe
