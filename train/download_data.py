import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm
import argparse

class ProgressPercentage(object):
    def __init__(self, total_size):
        self._total_size = total_size
        self._seen_so_far = 0
        self._lock = None
        self._pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        self._pbar.update(bytes_amount)

def download_data(args):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    bucket_name = args.bucket
    object_name = args.object
    local_file_path = args.file_path

    if not local_file_path:
        local_file_path = './data/' + object_name

    directory = os.path.dirname(local_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    meta_data = s3.head_object(Bucket=bucket_name, Key=object_name)
    total_size = int(meta_data.get('ContentLength', 0))

    s3.download_file(bucket_name, object_name, local_file_path, Callback=ProgressPercentage(total_size))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument('--bucket', type=str, default='edus-redpajama', help="bucket name")
    parser.add_argument('--object', type=str, default='slimpajama/train/tokenized/worker_1.bin', help="object name")
    parser.add_argument('--file_path', type=str, help="local file path to save")
    args = parser.parse_args()
    download_data(args)