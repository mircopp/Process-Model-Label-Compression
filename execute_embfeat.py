from stages.preprocess import preprocess
from stages.train import train
from save import save_preprocessed_data, load_preprocessed_data

if __name__ == '__main__':
    source_path = 'matrizes/embedding/'
    chunksize = 1
    try:
        X_google, y_google = load_preprocessed_data(chunksize=chunksize, source=source_path, name='preprocessed_X.google'), load_preprocessed_data(chunksize=chunksize, source=source_path, name='preprocessed_y.google')
    except FileNotFoundError:
        print('Preprocess embedded features.')
        X_google, y_google, X_pd, y_pd = preprocess(use_syn_feat=False)
        print('Save preprocessed data in chunks')
        save_preprocessed_data(X_google, source=source_path, name='preprocessed_X.google')
        save_preprocessed_data(y_google, source=source_path, name='preprocessed_y.google')
        save_preprocessed_data(X_pd, source=source_path, chunksize=1, name='preprocessed_X.pd')
        save_preprocessed_data(y_pd, source=source_path, chunksize=1, name='preprocessed_y.pd')
        print('Finished preprocessing.')

    print('Starting model training.')
    train(X_google, y_google, use_synfeat=False, use_crf_layer=False)
