import glob

import fire
import matplotlib
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.utils import resample
from jitl_data import _split_imageid, _get_videoid


def max_pooling_on_dataset(jit_data_file,
                           output_file,
                           mp_span_secs=1.0,
                           mp_stride_secs=0.5):
    """
    Run max pooling on a dataset's JITL input file and produce a smaller one
    :param dataset:
    :param base_dir:
    :param jit_data_file:
    :param mp_span_secs:
    :param mp_stride_secs:
    :return:
    """
    # if not isinstance(mp_span_secs, list):
    #     mp_span_secs = [mp_span_secs]
    # if not isinstance(mp_stride_secs, list):
    #     mp_stride_secs = [mp_stride_secs]

    df = pd.read_pickle(jit_data_file)
    print("Found {} images in total.".format(df.shape[0]))
    df['videoid'] = df['imageid'].map(lambda x: _get_videoid(x))
    df['frameid'] = df['imageid'].map(lambda x: _split_imageid(x)[1]).astype(int)
    df['grid_x'] = df['imageid'].map(lambda x: _split_imageid(x)[2]).astype(int)
    df['grid_y'] = df['imageid'].map(lambda x: _split_imageid(x)[3]).astype(int)

    span_frms = int(mp_span_secs * 30)
    stride_frms = int(mp_stride_secs * 30)
    print("Max pooling span frames={}, stride frame={}".format(span_frms, stride_frms))
    downsample_df = pd.DataFrame()

    video_id_grp = df.groupby(['videoid'])
    for video_id, video_rows in video_id_grp:
        print("Found {} frames for video {}".format(video_rows.shape[0], video_id))
        count = 0

        gridxy_grp = video_rows.groupby(['grid_x', 'grid_y'])
        for gridxy, inputs in gridxy_grp:
            inputs = inputs.sort_values(by=['frameid'])
            last_sent_imageid = None
            min_frm = inputs['frameid'].min()
            max_frm = inputs['frameid'].max()
            for pool_start_frm in range(min_frm, max_frm + 1, stride_frms):
                # print("Max pooling between frame {} and {}".format(pool_start_frm, pool_start_frm + span_frms))
                pool_images = inputs[(inputs['frameid'] >= pool_start_frm)
                                     & (inputs['frameid'] < pool_start_frm + span_frms)]

                dnn_scores = np.array(pool_images['prediction_proba'].tolist())[:, 1]
                assert dnn_scores.ndim == 1
                max_ind = np.argmax(dnn_scores)
                imageid = pool_images['imageid'].iloc[max_ind]
                if imageid != last_sent_imageid:
                    # print("sampled image: {}".format(imageid))
                    downsample_df = downsample_df.append(pool_images.iloc[max_ind], ignore_index=True)
                    last_sent_imageid = imageid
                    count += 1
        print("Sample {}/{} frames from video {}".format(count, video_rows.shape[0], video_id))

    downsample_df = downsample_df.sort_values(by=['imageid'])
    print("After max pooling, we have {} images".format(downsample_df.shape[0]))
    print("Sample 10 rows.")
    print downsample_df.iloc[::downsample_df.shape[0] / 10]

    if output_file:
        downsample_df.to_pickle(output_file)



class StealPositiveFromVideoEnd(object):
    def __init__(self, df, video_id, tail=10):
        super(StealPositiveFromVideoEnd, self).__init__()

        df = df[(df['videoid'] == video_id) & (df['label'].astype(bool))]
        df = df.sort_values(by=['frameid'])
        # print("Will steal these positives:")
        # print(df.iloc[-tail:])
        self.features = np.array(df.iloc[-tail:]['feature'].tolist())

    def __call__(self, n=5):
        samples = resample(self.features, n_samples=n, replace=False)
        return samples


def eval_jit_svm_on_dataset(jit_data_file,
                            output_file,
                            dnn_threshold_input_file=None,
                            dnn_cutoff_start=80, # dnn threshold for passing early discard filter
                            dnn_cutoff_end=100,
                            dnn_cutoff_step=2,
                            delta_t=10, # train every 10s
                            activate_threshold=5, # min number of examples per class needed to train the SVM,
                            # otherwise passthrough; training set is ever expanding;
                            svm_cutoff=0.3):
    if not isinstance(svm_cutoff, list):
        svm_cutoff = [svm_cutoff]
    if dnn_threshold_input_file is not None:
        print("Warning: Dnn_threshold_input_file is specified! Ignoring dnn_cutoff_start, dnn_cutoff_end, "
              "dnn_cutoff_step variable.")
        dnn_cutoff_list = np.load(dnn_threshold_input_file)
        dnn_cutoff_list.sort()
        print("loaded dnn cutoff threshold is: {}".format(dnn_cutoff_list))
    else:
        dnn_cutoff_list = [0.01 * x for x in range(dnn_cutoff_start, dnn_cutoff_end, dnn_cutoff_step)]
        print("Generated dnn cutoff list: {}".format(dnn_cutoff_list))
    df = pd.read_pickle(jit_data_file)
    print df.iloc[:5]
    df['videoid'] = df['imageid'].map(lambda x: _get_videoid(x))
    df['frameid'] = df['imageid'].map(lambda imgid: _split_imageid(imgid)[1]).astype(int)
    print df.iloc[:5]

    unique_videos = set(df['videoid'].tolist())
    result_df = pd.DataFrame()

    for video_id in unique_videos:
        for dnn_cutoff in dnn_cutoff_list:
            for svm_cut in svm_cutoff:
                print("-" * 50)
                print("Emulating video '{}' w/ DNN cutoff {}, SVM cutoff {}".format(video_id, dnn_cutoff, svm_cut))
                print("-" * 50)
                rv = run_once_jit_svm_on_video(df, video_id,
                                               dnn_cutoff=dnn_cutoff,
                                               delta_t=delta_t,
                                               activate_threshold=activate_threshold,
                                               svm_cutoff=svm_cut)
                result_df = result_df.append(rv, ignore_index=True)

    print result_df
    if output_file:
        result_df.to_pickle(output_file)


def run_once_jit_svm_on_video(df_in, video_id, dnn_cutoff,
                              delta_t=10, activate_threshold=5, svm_cutoff=0.3, augment_positive=False):
    # filter df by video id
    df = df_in[df_in['videoid'] == video_id]
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

    positive_supply = StealPositiveFromVideoEnd(df_in, video_id)

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

        X_jit = np.append(X_jit, X_test[sent_mask], axis=0)
        y_jit = np.append(y_jit, y_test[sent_mask], axis=0)
        assert X_jit.shape[1] == 1024
        # print("Found {} frames in window. Sent {}.".format(y_test.shape[0], np.count_nonzero(sent_mask)))

        # now, shall we (re-)train a new SVM?
        print("JIT training set {}/{}".format(y_jit.shape[0], np.count_nonzero(y_jit)))
        if np.count_nonzero(sent_mask) > 0 \
                and np.count_nonzero(y_jit == 0) >= activate_threshold \
                and (augment_positive or np.count_nonzero(y_jit == 1) >= activate_threshold):
            print("retraining")

            if not np.count_nonzero(y_jit == 1) >= activate_threshold and augment_positive:
                print("Houston, we don't have enough TPs.")
                augment_pos_X = positive_supply(n=activate_threshold)
                X_jit_train = np.append(X_jit, augment_pos_X, axis=0)
                y_jit_train = np.append(y_jit, np.ones((augment_pos_X.shape[0],)), axis=0)
                assert X_jit_train.shape[0] == y_jit_train.shape[0]
                print("Now you have {}/{}".format(y_jit_train.shape[0], np.count_nonzero(y_jit_train)))
            else:
                X_jit_train = X_jit
                y_jit_train = y_jit

            # use grid search to improve SVM accuracy
            # tuned_params = {
            #     'C': [1],
            #     'kernel': ['linear'],
            # }
            # clf = GridSearchCV(SVC(random_state=43,
            #                        max_iter=100,
            #                        class_weight='balanced',
            #                        probability=True,
            #                        verbose=True),
            #                    param_grid=tuned_params,
            #                    n_jobs=4,
            #                    refit=True)
            clf = SVC(random_state=42,
                      kernel='linear',
                      class_weight='balanced',
                      probability=True,
                      verbose=0)
            clf.fit(X_jit_train, y_jit_train)
        else:
            print("NOT retraining. Nothing new or not enough positives.")
            pass

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
                                    'label': y,
                                    'video_id': video_id,
                                    'svm_cutoff': svm_cutoff},
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
