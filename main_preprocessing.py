from stages.preprocess import preprocess
from util.disk_operations import save_preprocessed_data

if __name__ == '__main__':
    try:
        chunksize = 20
        source_path_syn = 'matrizes/embedding+syn/'

        print('Preprocess embedded+syn features.')
        X_google, y_google, X_pd, y_pd = preprocess(use_syn_feat=True)
        print('Save preprocessed data in chunks')
        save_preprocessed_data(X_google, source=source_path_syn, name='preprocessed_X.google')
        save_preprocessed_data(y_google, source=source_path_syn, name='preprocessed_y.google')
        save_preprocessed_data(X_pd, source=source_path_syn, chunksize=1, name='preprocessed_X.pd')
        save_preprocessed_data(y_pd, source=source_path_syn, chunksize=1, name='preprocessed_y.pd')

        source_path_emb = 'matrizes/embedding/'
        print('Preprocess embedded features.')
        X_google, y_google, X_pd, y_pd = preprocess(use_syn_feat=False)
        print('Save preprocessed data in chunks')
        save_preprocessed_data(X_google, source=source_path_emb, name='preprocessed_X.google')
        save_preprocessed_data(y_google, source=source_path_emb, name='preprocessed_y.google')
        save_preprocessed_data(X_pd, source=source_path_emb, chunksize=1, name='preprocessed_X.pd')
        save_preprocessed_data(y_pd, source=source_path_emb, chunksize=1, name='preprocessed_y.pd')
        print('Finished preprocessing.')
        ressources / google
    except FileNotFoundError:
        print('Tokenize sentences first before starting preprocessing!')
