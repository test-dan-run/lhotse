import os
import json
from argparse import ArgumentParser

from tqdm import tqdm
from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions
)
from lhotse.qa import remove_missing_recordings_and_supervisions

# returns {'recordings': recordings, 'supervisions': supervisions}

def convert_nemo_manifest(
    nemo_manifest_path: str,
    output_dir: str,
    language: str = None,
    ):

    basedir = os.path.dirname(nemo_manifest_path)

    with open(nemo_manifest_path, mode='r') as nemo_file:
        lines = nemo_file.readlines()
    items = [json.loads(line.rstrip('\r\n')) for line in lines]

    for item in tqdm(items, desc='Assigning recording id: '):
        item['recording_id'] = os.path.splitext(os.path.basename(item['audio_filepath']))[0]

    recordings = RecordingSet.from_recordings(
        Recording.from_file(
            os.path.join(basedir, item['audio_filepath']),
            recording_id = item['recording_id']
        )
        for item in items
    )

    supervisions = [
        SupervisionSegment(
            id = item['recording_id'],
            recording_id = item['recording_id'],
            language = language,
            start = 0,
            duration = recordings[item['recording_id']].duration,
            text = item['text'] if 'text' in item.keys() else None,
            speaker = item['speaker'] if 'speaker' in item.keys() else None,
            gender = item['gender'] if 'gender' in item.keys() else None,
        )
        for item in items
    ]
    supervisions = SupervisionSet.from_segments(supervisions)

    recordings, supervisions = remove_missing_recordings_and_supervisions(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        recordings.to_file(os.path.join(output_dir, 'recordings_all.jsonl.gz'))
        supervisions.to_file(os.path.join(output_dir, 'supervisions_all.jsonl.gz'))

    return {'recordings': recordings, 'supervisions': supervisions}

if __name__ == '__main__':
    parser = ArgumentParser(description='Converts nemo manifest into lhotse manifest format.')
    parser.add_argument('nemo_manifest_path', type=str)
    parser.add_argument('-o', '--output_dir', type=str, default=None, required=False)
    parser.add_argument('-l', '--language', type=str, default=None, required=False)

    args = parser.parse_args()
    convert_nemo_manifest(args.nemo_manifest_path, args.output_dir, args.language)