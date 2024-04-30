#/!bin/bash
for i in {1..32}
do
	python train/download_data.py --object [your_objects]${i}.bin --file_path ../../../data/worker_${i}.bin
done