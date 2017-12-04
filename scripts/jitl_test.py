import glob
import pickle

import fire
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from jitl_data import _split_imageid, _get_videoid

from result_analysis import datasets as result_datasets
from annotation_stats import dataset as dataset_stats


def eval_jit_svm_on_dataset(jit_data_file,
                            output_file,
                            dnn_cutoff_list=tuple([0.9 + 0.01 * x for x in range(0, 10)]),
                            delta_t=10,
                            activate_threshold=5):
    df = pd.read_pickle(jit_data_file)
    df['videoid'] = df['imageid'].map(lambda x: _get_videoid(x))
    df['frameid'] = df['imageid'].map(lambda imgid: _split_imageid(imgid)[1]).astype(int)
    print df.iloc[:5]

    unique_videos = set(df['videoid'].tolist())
    result_df = pd.DataFrame()

    for video_id in unique_videos:
        for dnn_cutoff in dnn_cutoff_list:
            print("-" * 50)
            print("Emulating video '{}' w/ DNN cutoff {}".format(video_id, dnn_cutoff))
            print("-" * 50)
            rv = run_once_jit_svm_on_video(df, video_id,
                                           dnn_cutoff=dnn_cutoff,
                                           delta_t=delta_t,
                                           activate_threshold=activate_threshold)
            result_df = result_df.append(rv, ignore_index=True)

    print result_df
    if output_file:
        result_df.to_pickle(output_file)


def run_once_jit_svm_on_video(df, video_id, dnn_cutoff, delta_t=10, activate_threshold=5, svm_cutoff=0.3):
    # filter df by video id
    df = df[df['videoid'] == video_id]
    # print df.iloc[0]

    dnn_proba = np.array(df['prediction_proba'].tolist())
    assert dnn_proba.shape[1] == 2, dnn_proba.shape
    dnn_fire = (dnn_proba[:, 1] >= dnn_cutoff)
    dnn_fire_index = np.nonzero(dnn_fire)[0]

    # filter df by dnn positive
    if len(dnn_fire_index) == 0:
        print("DNN fires nothing. Stop")
        return None

    print("DNN fires {} frames".format(len(dnn_fire_index)))
    df = df.iloc[dnn_fire_index]

    X = np.array(df['feature'].tolist())
    y = np.array(df['label'].tolist())
    imageids = df['imageid'].tolist()

    max_frame = df['frameid'].max()
    print("Max frame ID: {}".format(max_frame))

    X_jit = X[:0]  # cumulative, used to train JIT SVM
    y_jit = y[:0]  # same
    pred_jit = y[:0]  # store SVM's prediction on DNN's positive frames
    clf = None
    for t in range(0, int(1 + (max_frame / 30)), delta_t):
        # extract data within this window (from t to t+delta_t)
        # print("time window {} to {}".format(t, t + delta_t))
        df_test = df[(df['frameid'] >= t * 30) & (df['frameid'] < (t + delta_t) * 30)]
        # print df_test.iloc[:5]
        if df_test.empty:
            continue
        X_test = np.array(df_test['feature'].tolist())
        y_test = np.array(df_test['label'])
        assert X_test.shape[1] == 1024, str(X_test.shape)

        # Do we have an SVM to use?
        if clf:
            smv_proba = clf.predict_proba(X_test)
            predictions = (smv_proba[:, 1] >= svm_cutoff)
            # predictions = clf.predict(X_test)
        else:  # pass-through DNN's prediction (DNN says all are positive)
            predictions = np.ones_like(y_test)

        # write out to global prediction and cumulative JIT training set
        pred_jit = np.append(pred_jit, predictions, axis=0)
        sent_mask = (predictions == 1)

        X_jit = np.append(X_jit, X_test[sent_mask], axis=0)  # Hmm, stressing the RAM I know :-\
        y_jit = np.append(y_jit, y_test[sent_mask], axis=0)
        assert X_jit.shape[1] == 1024

        # print("Found {} frames in window. Sent {}.".format(y_test.shape[0], np.count_nonzero(sent_mask)))

        # now, shall we (re-)train a new SVM?
        # print("JIT samples {}/{}".format(y_jit.shape[0], np.count_nonzero(y_jit)))
        if clf or (np.count_nonzero(y_jit == 0) > activate_threshold
                   and np.count_nonzero(y_jit == 1) >= activate_threshold):
            # print("retraining")

            # use grid search to improve SVM accuracy
            tuned_params = {
                'C': [1, 10, 100],
                'kernel': ['linear'],
            }
            clf = GridSearchCV(SVC(random_state=42,
                                   max_iter=-1,
                                   class_weight='balanced',
                                   probability=True),
                               param_grid=tuned_params,
                               n_jobs=4,
                               refit=True)
            clf.fit(X_jit, y_jit)

    assert y.shape == pred_jit.shape, "y: {}, pred_jit: {}".format(y.shape, pred_jit.shape)
    assert y_jit.shape[0] == np.count_nonzero(pred_jit)
    jit_accuracy = accuracy_score(y, pred_jit)
    print("JIT accuracy: {}".format(jit_accuracy))

    res_df = pd.DataFrame().append({'delta_t': delta_t,
                                    'imageids': imageids,
                                    'dnn_cutoff': dnn_cutoff,
                                    'jitl_accuracy': jit_accuracy,
                                    'jitl_samples': y_jit.shape[0],
                                    'jitl_prediction': pred_jit,
                                    'video_id': video_id},
                                   ignore_index=True)
    print res_df
    return res_df


def plot_frame_accuracy(input_file, savefig=None):
    df = pd.read_csv(
        input_file,
        sep=r'\s+'
    )
    print df

    xlabels = map(int, df.columns[2:])
    for _, row in df.iterrows():
        x = xlabels
        y = np.array(row[2:])
        print x, y
        plt.plot(xlabels, np.array(row[2:]), '-o')

    plt.axis([0, max(xlabels), 0, 1.0])
    # plt.show()

    if savefig:
        plt.savefig(savefig)


def plot_rolling_svm(file_glob, savefig=None):
    paths = glob.glob(file_glob)
    df = pd.DataFrame()
    for path in paths:
        print("Parsing {}".format(path))
        df1 = pd.read_csv(path, sep=' ')
        df = df.append(df1, ignore_index=True)

    print df

    # df = df[df['delta_t'] < 90]
    video_ids = set(df['video_id'])

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("$\Delta t$ (sec)")
    ax1.set_ylabel("Frame accuracy")
    ax1.set_ylim((0, 1))
    ax2 = ax1.twinx()
    ax2.set_ylabel("# frames transmitted")
    # plt.xticks(sorted(set(df['delta_t'])), sorted(set(df['delta_t'])))
    for vid in video_ids:
        df_video = df[df['video_id'] == vid]
        # accuracy
        ax1.plot(df_video['delta_t'], df_video['jit_accuracy'], '-')
        ax2.plot(df_video['delta_t'], df_video['jit_samples'], '--')

    if savefig:
        plt.savefig(savefig)

    plt.show()


if __name__ == '__main__':
    fire.Fire()
