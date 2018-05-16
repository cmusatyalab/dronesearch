import collections
import io_util

okutama_video_to_frame_num = {
    '1.1.10': 2630,
    '1.1.11': 604,
    '1.1.1': 2272,
    '1.1.2': 2220,
    '1.1.3': 1966,
    '1.1.4': 1950,
    '1.1.5': 1560,
    '1.1.6': 2381,
    '1.1.7': 2519,
    '1.2.11': 1810,
    '1.2.2': 1098,
    '1.2.4': 1973,
    '1.2.5': 1026,
    '1.2.6': 1014,
    '1.2.7': 1837,
    '1.2.8': 1541,
    '1.2.9': 1373,
    '2.1.10': 2713,
    '2.1.1': 1235,
    '2.1.2': 1398,
    '2.1.3': 2878,
    '2.1.4': 2108,
    '2.1.5': 1825,
    '2.1.6': 2519,
    '2.1.7': 2514,
    '2.2.11': 1285,
    '2.2.2': 1466,
    '2.2.4': 2029,
    '2.2.5': 1042,
    '2.2.6': 2254,
    '2.2.7': 1530,
    '2.2.8': 1770,
    '2.2.9': 1502
}
okutama_original_width, okutama_original_height = 3840, 2160
okutama_train_videos = [
    '1.1.3', '1.1.2', '1.1.5', '1.1.4', '2.2.7', '2.1.7', '2.1.4', '2.2.11',
    '1.1.10'
]
okutama_test_videos = ['2.2.2', '2.2.4', '1.1.7']
okutama_video_id_to_original_resolution = collections.defaultdict(
    lambda: (okutama_original_width, okutama_original_height))

stanford_video_to_frame_num = {
    'bookstore_video0': 13335,
    'bookstore_video1': 14558,
    'bookstore_video2': 14558,
    'bookstore_video3': 14558,
    'bookstore_video4': 14558,
    'bookstore_video5': 14558,
    'bookstore_video6': 14558,
    'coupa_video0': 11966,
    'coupa_video1': 11966,
    'coupa_video2': 11966,
    'coupa_video3': 11966,
    'deathCircle_video0': 12721,
    'deathCircle_video1': 14065,
    'deathCircle_video2': 431,
    'deathCircle_video3': 12492,
    'deathCircle_video4': 452,
    'gates_video0': 9006,
    'gates_video1': 9006,
    'gates_video2': 9006,
    'gates_video3': 9199,
    'gates_video4': 2202,
    'gates_video5': 2082,
    'gates_video6': 2082,
    'gates_video7': 2202,
    'gates_video8': 2202,
    'hyang_video0': 11368,
    'hyang_video10': 9928,
    'hyang_video11': 9928,
    'hyang_video12': 9928,
    'hyang_video13': 9928,
    'hyang_video14': 9928,
    'hyang_video1': 13438,
    'hyang_video2': 12272,
    'hyang_video3': 12272,
    'hyang_video4': 8057,
    'hyang_video5': 10648,
    'hyang_video6': 9928,
    'hyang_video7': 574,
    'hyang_video8': 574,
    'hyang_video9': 574,
    'little_video0': 1518,
    'little_video1': 14070,
    'little_video2': 14070,
    'little_video3': 14070,
    'nexus_video0': 12681,
    'nexus_video10': 11472,
    'nexus_video11': 11472,
    'nexus_video1': 12681,
    'nexus_video2': 12681,
    'nexus_video3': 1062,
    'nexus_video4': 1062,
    'nexus_video5': 1062,
    'nexus_video6': 12016,
    'nexus_video7': 12016,
    'nexus_video8': 12016,
    'nexus_video9': 11472,
    'quad_video0': 509,
    'quad_video1': 509,
    'quad_video2': 509,
    'quad_video3': 509
}
stanford_train_videos = [
    "little_video1", "nexus_video1", "hyang_video5", "little_video3",
    "gates_video5", "gates_video6", "gates_video0", "gates_video3",
    "coupa_video2", "coupa_video1", "deathCircle_video0", "bookstore_video2",
    "bookstore_video3", "bookstore_video0", "bookstore_video1",
    "deathCircle_video3"
]

stanford_test_horizontal_videos = [
    "bookstore_video5", "bookstore_video4"
]

stanford_test_vertical_videos = [
    "little_video2", "hyang_video4",
    "gates_video1"
]

stanford_test_videos = (stanford_test_horizontal_videos
                        + stanford_test_vertical_videos)

stanford_video_id_to_original_resolution = {
    "quad_video0": (1983, 1088),
    "hyang_video8": (1350, 1940),
    "deathCircle_video4": (1452, 1994),
    "quad_video3": (1983, 1088),
    "quad_video1": (1983, 1088),
    "quad_video2": (1983, 1088),
    "nexus_video11": (1311, 1980),
    "nexus_video10": (1311, 1980),
    "nexus_video2": (1330, 1947),
    "hyang_video11": (1416, 748),
    "hyang_video10": (1416, 748),
    "hyang_video13": (1316, 748),
    "hyang_video12": (1316, 848),
    "hyang_video14": (1316, 748),
    "nexus_video6": (1331, 1962),
    "gates_video8": (1334, 1982),
    "deathCircle_video2": (1436, 1959),
    "hyang_video9": (1350, 1940),
    "deathCircle_video1": (1409, 1916),
    "nexus_video5": (1184, 1759),
    "deathCircle_video3": (1400, 1904),
    "little_video1": (1322, 1945),
    "little_video0": (1417, 2019),
    "little_video3": (1422, 1945),
    "nexus_video0": (1330, 1947),
    "hyang_video1": (1445, 2002),
    "nexus_video3": (1184, 1759),
    "hyang_video3": (1433, 741),
    "hyang_video2": (1433, 841),
    "hyang_video5": (1454, 1991),
    "hyang_video4": (1340, 1730),
    "nexus_video9": (1411, 1980),
    "nexus_video8": (1331, 1962),
    "nexus_video1": (1430, 1947),
    "little_video2": (1322, 1945),
    "gates_video4": (1434, 1982),
    "gates_video5": (1426, 2011),
    "gates_video6": (1326, 2011),
    "gates_video7": (1334, 1982),
    "gates_video0": (1325, 1973),
    "gates_video1": (1425, 1973),
    "gates_video2": (1325, 1973),
    "gates_video3": (1432, 2002),
    "coupa_video2": (1980, 1093),
    "coupa_video3": (1980, 1093),
    "coupa_video0": (1980, 1093),
    "coupa_video1": (1980, 1093),
    "deathCircle_video0": (1630, 1948),
    "bookstore_video6": (1322, 1079),
    "bookstore_video4": (1322, 1079),
    "bookstore_video5": (1322, 1079),
    "bookstore_video2": (1422, 1079),
    "bookstore_video3": (1322, 1079),
    "bookstore_video0": (1424, 1088),
    "bookstore_video1": (1422, 1079),
    "hyang_video0": (1455, 1925),
    "hyang_video7": (1450, 1940),
    "nexus_video7": (1431, 1962),
    "hyang_video6": (1416, 848),
    "nexus_video4": (1284, 1759)
}

elephant_video_id_to_original_resolution = {
    "01": (1280, 720),
    "02": (1280, 720),
    "03": (1280, 720),
    "04": (1280, 720),
    "05": (1280, 720),
    "06": (1280, 720),
    "07": (1280, 720),
    "08": (960, 720),
    "09": (1280, 720),
    "10": (1280, 720),
    "11": (1280, 720)
}

elephant_video_id_to_tpod_resolution = {
    "01": (720, 404),
    "02": (720, 404),
    "03": (720, 404),
    "04": (720, 404),
    "05": (720, 404),
    "06": (720, 404),
    "07": (720, 404),
    "08": (640, 480),
    "09": (720, 404),
    "10": (720, 404),
    "11": (720, 404)
}

elephant_video_id_to_frame_num = {
    "01": 1248,
    "02": 196,
    "03": 375,
    "04": 21018,
    "05": 3454,
    "06": 6307,
    "07": 2183,
    "08": 4685,
    "09": 9369,
    "10": 899,
    "11": 4469
}

elephant_train_videos = sorted(
    elephant_video_id_to_original_resolution.keys())[:8]
elephant_test_videos = sorted(
    elephant_video_id_to_original_resolution.keys())[8:]

raft_video_id_to_original_resolution = {
    "01": (1280, 720),
    "02": (1280, 720),
    "03": (1280, 720),
    "04": (1280, 720),
    "05": (1280, 720),
    "06": (1280, 720),
    "07": (1280, 720),
    "08": (1280, 720),
    "09": (1280, 720),
    "10": (1280, 720),
    "11": (1280, 720)
}

raft_video_id_to_tpod_resolution = {
    "01": (720, 404),
    "02": (720, 404),
    "03": (720, 404),
    "04": (720, 404),
    "05": (720, 404),
    "06": (720, 404),
    "07": (720, 404),
    "08": (720, 404),
    "09": (720, 404),
    "10": (720, 404),
    "11": (720, 404)
}

raft_video_id_to_frame_num = {
    "01": 6600,
    "02": 2938,
    "03": 2875,
    "04": 5469,
    "05": 968,
    "06": 7534,
    "07": 8994,
    "08": 7639,
    "09": 5947,
    "10": 2783,
    "11": 2648
}

raft_train_videos = sorted(raft_video_id_to_original_resolution.keys())[:8]
raft_test_videos = sorted(raft_video_id_to_original_resolution.keys())[8:]
dataset = {
    "elephant": {
        'annotation_func':
        io_util.load_stanford_campus_annotation,
        'video_id_to_original_resolution':
        elephant_video_id_to_original_resolution,
        'video_id_to_frame_num':
        elephant_video_id_to_frame_num,
        'labels': ['elephant'],
        'video_ids':
        elephant_train_videos + elephant_test_videos,
        'train':
        elephant_train_videos,
        'test':
        elephant_test_videos,
        'extra_negative_dataset':
        ['okutama', 'stanford', 'raft'],
        'total_test_frames': 92378,
    },
    "raft": {
        'annotation_func':
        io_util.load_stanford_campus_annotation,
        'video_id_to_original_resolution':
        raft_video_id_to_original_resolution,
        'video_id_to_frame_num': raft_video_id_to_frame_num,
        'labels': ['raft'],
        'video_ids': raft_train_videos + raft_test_videos,
        'train': raft_train_videos,
        'test': raft_test_videos,
        'extra_negative_dataset':
        ['okutama', 'stanford', 'elephant'],
        'total_test_frames': 92378,
    },
    "okutama": {
        'annotation_func':
        io_util.load_okutama_annotation,
        'video_id_to_original_resolution':
        okutama_video_id_to_original_resolution,
        'video_id_to_frame_num':
        okutama_video_to_frame_num,
        'labels': ['Person'],
        'video_ids':
        okutama_train_videos + okutama_test_videos,
        'train':
        okutama_train_videos,
        'test':
        okutama_test_videos,
        'extra_negative_dataset':
        ['elephant'],
        'total_test_frames': 20751,
    },
    "stanford": {
        'annotation_func':
        io_util.load_stanford_campus_annotation,
        'video_id_to_original_resolution':
        stanford_video_id_to_original_resolution,
        'video_id_to_frame_num':
        stanford_video_to_frame_num,
        'labels': ['Car', 'Bus'],
        'video_ids':
        stanford_train_videos + stanford_test_videos,
        'train':
        stanford_train_videos,
        'test':
        stanford_test_videos,
        'test_horizontal':
        stanford_test_horizontal_videos,
        'test_vertical':
        stanford_test_vertical_videos,
        'extra_negative_dataset':
        ['okutama', 'raft', 'elephant'],
        'total_test_frames': 92378,
    }
}
