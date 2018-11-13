import json
from stages.tokenization import annotate_data, build_matrizes, build_csv_matrix

def read_google_data(data_source):
    data_string = open(data_source, 'rb').read().decode('utf-8')
    sentences = data_string.split('\n\n')
    print('Number of sentences:', len(sentences))
    json_objects = []
    for chunk in sentences:
        try:
            obj = json.loads(chunk)
            current = {'sentence': obj['graph']['sentence'], 'compression': obj['compression']['text']}
            json_objects.append(current)
        except json.decoder.JSONDecodeError:
            print('Found decode error:', chunk)
            continue
    return json_objects

def read_process_description_data(data_source):
    data_string = open(data_source, 'rb').read().decode('utf-8')
    json_objects = json.loads(data_string)
    print('Number of sentences:', len(json_objects))
    return json_objects

if __name__ == '__main__':
    try:
        # Google corpus
        GOOGLE_DATA_SOURCES = ['sent-comp.train01.json', 'sent-comp.train02.json',
                               'sent-comp.train03.json', 'sent-comp.train04.json',
                               'sent-comp.train05.json', 'sent-comp.train06.json',
                               'sent-comp.train07.json', 'sent-comp.train08.json',
                               'sent-comp.train09.json', 'sent-comp.train10.json',
                               'comp-data.eval.json']
        for data_source in GOOGLE_DATA_SOURCES:
            print('Reading ', data_source)
            data = read_google_data('ressources/google' + data_source)
            sentences, compressions = annotate_data(data)
            X = build_matrizes(sentences, compressions)

            header_row = ['word', 'pos', 'dependency_label', 'id', 'parent', 'EOS', 'kept']
            data = build_csv_matrix(header_row, X)
            data.to_csv('csv/google/' + data_source + '.csv', sep=',', encoding='utf-8', na_rep='N/A')
            print('Totally saved:', len(X))

        # Process description corpus
        PD_DATA_SOURCES = ['process1.json', 'process2.json', 'process3.json']
        for data_source in PD_DATA_SOURCES:
            print('Reading ', data_source)
            data = read_process_description_data('ressources/process_descriptions/' + data_source)
            sentences, compressions = annotate_data(data)
            X = build_matrizes(sentences, compressions)

            header_row = ['word', 'pos', 'dependency_label', 'id', 'parent', 'EOS', 'kept']
            data = build_csv_matrix(header_row, X)
            data.to_csv('csv/process_descriptions/' + data_source + '.csv', sep=',', encoding='utf-8', na_rep='N/A')
            print('Totally saved:', len(X))
    except FileNotFoundError:
        print('Download full corpus first and add it to ressources/process_descriptions and ressources/google first')
