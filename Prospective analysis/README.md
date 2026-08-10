## Prospective analysis results

The results of the prospective analysis are contained in the CSV file. Video names are reported in the `filename_video` column, and the corresponding JSON file containing the predictions for each video is reported in the `source_file` column.

The source_file_per_video` column repeats the JSON filename for every video. Predictions for each video are reported in `predicted_class_per_video`, while true values for each video are reported in `true_class_per_video`. Finally, the prediction per label is reported in `predicted_class`, and the true label is reported in `true_class`. The `Count_per_video` and `count_per_prediction` columns report 1 if the prediction is correct and 0 if it is incorrect; all evaluation metrics are derived from these two columns.

Given the limited amount of data, the results were also double-checked manually.

See the Supporting Information for further details.

