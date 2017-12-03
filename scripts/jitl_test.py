import glob
import pickle

import fire
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils import resample

from scripts.jitl_data import load_pre_logit_Xy, _split_imageid


def visualize(pre_logit_files):
    # visualize the temporal locality of event frames
    X, y, _ = load_pre_logit_Xy(pre_logit_files)
    print ''.join('*' if t == 1 else '-' for t in y)


# CLASSIFIER_CLS = SVC
CLASSIFIER_CLS = SGDClassifier


def emulate_rolling_svm(pre_logit_files, delta_ts, activate_threshold=50, verbose=False,
                        save_result=None, save_inference_result=None):
    """

    :param pre_logit_files:
    :param delta_t: retrain a new SVM every delta_t seconds
    :param activate_threshold: min samples per class to activate SVM
    :return:
    """
    X, y, sorted_imageids = load_pre_logit_Xy(pre_logit_files)
    assert X.shape[0] == y.shape[0]

    df = pd.DataFrame({'imageid': sorted_imageids, 'label': y, 'feature': X.tolist()})
    df['frameid'] = df['imageid'].map(lambda imgid: _split_imageid(imgid)[1])

    # print(df.iloc[::df.shape[0] / 10, :])
    video_id, _, _, _ = _split_imageid(sorted_imageids[0])
    print("Video ID: {}".format(video_id))
    max_frame = df['frameid'].max()
    print("Max frame: {}".format(max_frame))

    result_df = pd.DataFrame()
    baseline_precision = float(np.count_nonzero(y)) / y.shape[0]
    print("Initial precision: {}".format(baseline_precision))
    result_df = result_df.append({'delta_t': 0,
                                  'jit_accuracy': baseline_precision,
                                  'jit_samples': y.shape[0],
                                  'video_id': video_id,
                                  'inference_result': np.stack([y, 1 - y], axis=1)},
                                 ignore_index=True)

    if not isinstance(delta_ts, list):
        delta_ts = [delta_ts]

    for delta_t in delta_ts:
        if delta_t * 30 > max_frame:
            break
        X_jit = X[:0]
        y_jit = y[:0]  # used to train JIT SVM
        pred_jit = y[:0]  # used to evaluate end-to-end accuracy
        clf = None
        inference_result = dict()   # { imageid -> list(float) }
        print("delta_t={}".format(delta_t))
        for t in range(0, 1 + (max_frame / 30), delta_t):
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
                predictions = clf.predict(X_test)
            else:  # pass-through DNN's prediction (DNN says all are positive)
                predictions = np.ones_like(y_test)

            # write out to global prediction and cumulative JIT training set
            pred_jit = np.append(pred_jit, predictions, axis=0)

            sent_mask = (predictions == 1)
            X_jit = np.append(X_jit, X_test[sent_mask], axis=0)  # Hmm, stressing the RAM I know :-\
            y_jit = np.append(y_jit, y_test[sent_mask], axis=0)
            assert X_jit.shape[1] == 1024

            # store prediction proba for event recall analysis
            imageids = df_test['imageid'].tolist()
            assert len(imageids) == y_test.shape[0]
            batch_result = clf.predict_proba(X_test).tolist() if clf \
                else np.stack([predictions, 1 - predictions], axis=1).tolist()
            inference_result.update(zip(imageids, batch_result))

            print("Found {} frames in window. Sent {}.".format(y_test.shape[0], np.count_nonzero(sent_mask)))

            # now, shall we (re-)train a new SVM?
            print("JIT samples {}/{}".format(y_jit.shape[0], np.count_nonzero(y_jit)))
            if clf or (np.count_nonzero(y_jit == 0) > activate_threshold
                       and np.count_nonzero(y_jit == 1) >= activate_threshold):
                print("retraining")

                # use grid search to improve SVM accuracy
                tuned_params = {
                    'C': [1, 10, 100],
                    'kernel': ['linear'],
                }
                clf = GridSearchCV(SVC(random_state=42,
                                       max_iter=-1,
                                       verbose=verbose,
                                       class_weight='balanced',
                                       probability=True),
                                   param_grid=tuned_params,
                                   n_jobs=4,
                                   refit=True)
                clf.fit(X_jit, y_jit)

        assert y.shape == pred_jit.shape, "y: {}, pred_jit: {}".format(y.shape, pred_jit.shape)
        jit_accuracy = accuracy_score(y, pred_jit)
        print("JIT accuracy: {}".format(jit_accuracy))

        result_df = result_df.append({'delta_t': delta_t,
                                      'jit_accuracy': jit_accuracy,
                                      'jit_samples': y_jit.shape[0],
                                      'video_id': video_id,
                                      'inference_result': inference_result},
                                     ignore_index=True)

    print result_df

    if save_result:
        result_df.to_csv(save_result, sep=' ')

    if save_inference_result:
        # save inference result of the best delta_t
        best_inference_result = result_df.loc[result_df['jit_accuracy'].idxmax()]['inference_result']
        assert isinstance(best_inference_result, dict)
        pickle.dump(best_inference_result,
                    open(save_inference_result, 'wb'))


def train(pre_logit_files,
          save_model_path=None,
          test_ratio=0.1,
          split_pos=True,
          downsample_train=1.0,
          verbose=False):
    """
    :param downsample_train: down sample the training split.
    :param split_pos: if true, we split the data according to the ratio of positive/negative separately,
        instead of ratio of all samples.
    :param pre_logit_files:
    :param save_model_path:
    :param eval_every_iters:
    :param n_iters:
    :param test_ratio:
    :param shuffle: If false, sample order is retained when splitting.
    :param verbose:
    :return:
    """
    X, y, _ = load_pre_logit_Xy(pre_logit_files)
    assert X.shape[0] == y.shape[0]
    n_all = y.shape[0]

    if not split_pos:
        n_test = int(n_all * test_ratio)
        X_train, X_test = X[: -n_test], X[-n_test:]
        y_train, y_test = y[: -n_test], y[-n_test:]
    else:
        print "Splitting train/validation for positive/negative respectively."
        X_pos, y_pos = X[y == 1], y[y == 1]
        X_neg, y_neg = X[y == 0], y[y == 0]

        n_test_pos = int(X_pos.shape[0] * test_ratio)
        n_test_neg = int(X_neg.shape[0] * test_ratio)

        X_pos_train, X_pos_test = X_pos[: -n_test_pos], X_pos[-n_test_pos:]
        y_pos_train, y_pos_test = y_pos[: -n_test_pos], y_pos[-n_test_pos:]

        X_neg_train, X_neg_test = X_neg[: -n_test_neg], X_neg[-n_test_neg:]
        y_neg_train, y_neg_test = y_neg[: -n_test_neg], y_neg[-n_test_neg:]

        X_train = np.concatenate([X_pos_train, X_neg_train], axis=0)
        y_train = np.concatenate([y_pos_train, y_neg_train])

        X_test = np.concatenate([X_pos_test, X_neg_test], axis=0)
        y_test = np.concatenate([y_pos_test, y_neg_test])

    # Downsample training set
    n_train = int(X_train.shape[0] * downsample_train)
    if not split_pos:
        X_train, y_train = resample(X_train, y_train, n_samples=n_train, random_state=42)
    else:
        X_pos_train, y_pos_train = resample(X_pos_train, y_pos_train, n_samples=max(n_train / 2, 1), random_state=42)
        X_neg_train, y_neg_train = resample(X_neg_train, y_neg_train, n_samples=max(n_train / 2, 1), random_state=42)
        X_train = np.concatenate([X_pos_train, X_neg_train], axis=0)
        y_train = np.concatenate([y_pos_train, y_neg_train])

    assert X_train.shape[1] == X_test.shape[1] == 1024
    print "All: %d / %d" % (y.shape[0], np.count_nonzero(y))
    print "Train set: %d / %d" % (y_train.shape[0], np.count_nonzero(y_train))
    print "Test set: %d / %d" % (y_test.shape[0], np.count_nonzero(y_test))

    clf = CLASSIFIER_CLS(random_state=42,
                         verbose=verbose,
                         class_weight='balanced')

    clf.fit(X_train, y_train)

    print "Final train accuracy: %f" % clf.score(X_train, y_train)
    print "Final test accuracy: %f " % clf.score(X_test, y_test)

    print "Confusion matrix on test:"
    pred_test = clf.predict(X_test)
    cm = confusion_matrix(y_true=y_test, y_pred=pred_test)
    print cm

    if save_model_path is not None:
        print "saving model to " + save_model_path
        pickle.dump(clf, open(save_model_path, 'wb'))


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
