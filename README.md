# License Plate Recognition pipeline
This pipeline takes as an input a video and outputs a CSV file containing the license plates that were recognized in the video at certain frames and timestamps.

# How to run the whole pipeline
You can run the whole pipeline by running the file evaluator.sh.

This however fails on some devices so below you can find the instructions on how to run main.py and evaluation.py separately.

There are also some comments in evaluator.sh on how you could possibly fix the issues for your device but we recommend following the steps below instead.

# How to run main.py
You can get an output file by running the following command:
    python main.py --file_path <path_to_input_video> --output_path <path_to_output_file> --sample_frequency <sample_frequency>

The default value for --file_path is 'dataset/TrainingsVideo.avi'

If no value is specified for --output_path, it will be outputted to 'Output.csv'.

The default value for the sample frequency is 2. It is not advised to use other values as it decreases the accuracy of our pipeline. You could change it to 1 but it makes the execution time longer.

After running this command there should be a file created at <path_to_output_file> if there were any license plates detected in the video.

# How to run evaluation.py
You can evaluate the performance of the algorithm by running the following command:
    python evaluation.py --file_path <path_to_csv> --ground_truth_path <path_to_groundtruth>

Here is an example of the command when the output file is located at Output.csv:
    python evaluation.py --file_path Output.csv --ground_truth_path dataset/groundTruth.csv

After running this command you should get an overview of the performance of our algorithm for each category.