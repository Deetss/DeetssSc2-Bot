run:
	python3 run.py

model:
	CUDA_VISIBLE_DEVICES="" python3 model.py

observe:
	python3 observer.py

test:
	pytest