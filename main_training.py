from stages.train import train
from util.disk_operations import load_preprocessed_data

if __name__ == '__main__':
    source_path_syn = 'matrizes/embedding+syn/'
    source_path_emb = 'matrizes/embedding/'
    chunksize = 20

    try:
        X_google, y_google = load_preprocessed_data(chunksize=chunksize, source=source_path_syn, name='preprocessed_X.google'), load_preprocessed_data(chunksize=chunksize, source=source_path_syn, name='preprocessed_y.google')
        print('Starting syn model training.')
        train(X_google, y_google, use_synfeat=True)

        X_google, y_google = load_preprocessed_data(chunksize=chunksize, source=source_path_emb,
                                                        name='preprocessed_X.google'), load_preprocessed_data(
                chunksize=chunksize, source=source_path_emb, name='preprocessed_y.google')

        print('Starting emb model training.')
        train(X_google, y_google, use_synfeat=False)

    except FileNotFoundError:
        print('Execute preprocessing first!')


