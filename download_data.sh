#/!bin/bash
for i in {1..32}
do
	python train/download_data.py --object slimpajama/train/tokenized/worker_${i}.bin --file_path ../../../data/worker_${i}.bin
done