# iccv2019-learning-to-drive
ICCV 2019: Learning-to-Drive Challenge


## Scripts

#### Create submission script

```bash
usage: create_submission.py [-h] --input-path INPUT_PATH --output-path
                            OUTPUT_PATH --original-dataset-path
                            ORIGINAL_DATASET_PATH --sampled-dataset-path
                            SAMPLED_DATASET_PATH --frequency FREQUENCY
                            --number NUMBER

optional arguments:
  -h, --help            show this help message and exit
  --input-path INPUT_PATH, -i INPUT_PATH
                        Path where to read the input submission for processing
  --output-path OUTPUT_PATH, -o OUTPUT_PATH
                        Path where to save the output submission
  --original-dataset-path ORIGINAL_DATASET_PATH, -od ORIGINAL_DATASET_PATH
                        Path to original dataset csv describing frames
  --sampled-dataset-path SAMPLED_DATASET_PATH, -sd SAMPLED_DATASET_PATH
                        Path to sampled dataset csv describing frames
  --frequency FREQUENCY, -f FREQUENCY
                        With what frequency the dataset has been loaded for
                        test prediction
  --number NUMBER, -n NUMBER
                        What number of previous frames has been loaded for
                        test prediction
```
