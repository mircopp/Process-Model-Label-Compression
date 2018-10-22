import json
from compression.tokenization import annotate_data, build_matrizes, build_csv_matrix

def read_process_description_data(data_source):
    data_string = open(data_source, 'rb').read().decode('utf-8')
    json_objects = json.loads(data_string)
    print('Number of sentences:', len(json_objects))
    return json_objects

if __name__ == '__main__':
    # Process description corpus
    PD_DATA_SOURCES = ['process1.json', 'process2.json', 'process3.json']
    for data_source in PD_DATA_SOURCES:
        print('Reading ', data_source)
        data = read_process_description_data('Ressources/process_descriptions/' + data_source)
        sentences, compressions = annotate_data(data)
        X = build_matrizes(sentences, compressions)

        header_row = ['word', 'pos', 'dependency_label', 'id', 'parent', 'EOS', 'kept']
        data = build_csv_matrix(header_row, X)
        data.to_csv('csv/process_descriptions/' + data_source + '.csv', sep=',', encoding='utf-8', na_rep='N/A')
        print('Totally saved:', len(X))
