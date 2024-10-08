TRAIN_SCRIPT = ./train.sh
INFERENCE_SCRIPT = ./inference.sh
CONVERT_SCRIPT = ./convert.sh
DOWNLOAD_SCRIPT = ./download.sh

TRAIN_CONFIG_PATH ?= configs/train_cfg_1.yaml
TRAIN_NEW_CONFIG_PATH ?= configs/train_cfg_2.yaml
INFERENCE_CONFIG_PATH ?= configs/inference_cfg_1.yaml
INFERENCE_NEW_CONFIG_PATH ?= configs/inference_cfg_2.yaml

TRAIN_CSV_PATH ?= data/train.csv
TEST_CSV_PATH ?= data/test.csv
EXAMPLE_CSV_PATH ?= data/example.csv

OUTPUT_NAME ?= output.npy
NEW_OUTPUT_NAME ?= output_new.npy
PROBS_PATH ?= $(OUTPUT_NAME)
PROBS_NEW_PATH ?= $(NEW_OUTPUT_NAME)
OUTPUT_CSV ?= submission.csv

.PHONY: train train_new
train:
	@echo "Starting training with config: $(TRAIN_CONFIG_PATH)..."
	@$(TRAIN_SCRIPT) $(TRAIN_CONFIG_PATH) $(TRAIN_CSV_PATH)

train_new:
	@echo "Starting training with new config: $(TRAIN_NEW_CONFIG_PATH)..."
	@$(TRAIN_SCRIPT) $(TRAIN_NEW_CONFIG_PATH) $(TRAIN_CSV_PATH)

.PHONY: inference inference_new
inference:
	@echo "Starting inference with config: $(INFERENCE_CONFIG_PATH)..."
	@$(INFERENCE_SCRIPT) $(INFERENCE_CONFIG_PATH) $(TEST_CSV_PATH) $(OUTPUT_NAME)

inference_new:
	@echo "Starting inference with new config: $(INFERENCE_NEW_CONFIG_PATH)..."
	@$(INFERENCE_SCRIPT) $(INFERENCE_NEW_CONFIG_PATH) $(TEST_CSV_PATH) $(NEW_OUTPUT_NAME)

.PHONY: convert
convert:
	@echo "Converting predictions..."
	@$(CONVERT_SCRIPT) $(EXAMPLE_CSV_PATH) $(PROBS_PATH) $(PROBS_NEW_PATH) $(OUTPUT_CSV)

.PHONY: clean
clean:
	@echo "Cleaning output files..."
	@rm -f *.npy submission.csv
	@echo "Cleaned up!"

.PHONY: download
download:
	@echo "Downloading model weights..."
	@$(DOWNLOAD_SCRIPT)