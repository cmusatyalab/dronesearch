#!/usr/bin/env bash

die() { echo "$@" 1>&2 ; exit 1; }


ORIG_TEST_DIR="/home/junjuew/mobisys18/processed_dataset/munich/mobilenet_test/"
PICK="4K0G0110"

echo ""
echo "Pick: ${PICK} from ${ORIG_TEST_DIR}"
echo "We must pick one picture from the original test set such that the pre-trained network has never seen."
echo ""

# Put "train" in directory name to facilitate inference in download_and_convert_data.py
DEST_DIR="/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_jit_train"

if [[ -d "${DEST_DIR}/photos" ]]; then
    die "${DEST_DIR}/photos already exists. Quiting."
fi

mkdir -p ${DEST_DIR}/photos/positive ${DEST_DIR}/photos/negative

echo "Linking image files to ${DEST_DIR}/photos"

ln -s ${ORIG_TEST_DIR}/photos/positive/*${PICK}* ${DEST_DIR}/photos/positive/
ln -s ${ORIG_TEST_DIR}/photos/negative/*${PICK}* ${DEST_DIR}/photos/negative/

echo "Done"
