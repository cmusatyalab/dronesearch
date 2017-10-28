import pandas as pd


def write_list_to_file(input_list, file_path, delimiter='\n'):
    with open(file_path, "w") as output:
        output.write(delimiter.join(input_list))


def parse_annotation_file(annotation_file_path):
    annotations = pd.read_csv(annotation_file_path,
                              sep=' ',
                              header=None,
                              names=['trackid', 'xmin', 'ymin', 'xmax',
                                     'ymax', 'frameid', 'lost', 'occluded',
                                     'generated', 'label'])
    return annotations


#parse_annotation_file('nexus/video1/annotations.txt')
