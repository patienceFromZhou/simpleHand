train:
	torchrun --nproc_per_node 8 train.py --resume

eval:
	python3 infer_to_json.py epoch_200
