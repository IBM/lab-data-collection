## Prospective analysis results

The CSV file contains the prospective analysis results. The `filename_video` column lists each video name, and `source_file` gives the corresponding JSON filename containing predictions for that video.

The `source_file_per_video` column repeats the JSON filename for each video. Predictions and true values for each video are listed in `predicted_class_per_video` and `true_class_per_video`, respectively. Predictions and true labels are listed in `predicted_class` and `true_class`, respectively. The `Count_per_video` and `count_per_prediction` columns contain 1 when a prediction is correct and 0 otherwise. All evaluation metrics are derived from these columns.

Because the dataset is limited, the results were also checked manually.

See the Supporting Information for further details. The `uv.lock` file pins the complete transitive dependency graph.

