# Data preparation and Model training

When the system is used to acquire data, couples of videos and labels are created. However, prior utilization in the training, polishing and preparation are required.
The following steps should be followed in the order:

<ol>
<li>Using the script <code>VideoLabelCounter.py</code>, the number of labels and the number of videos are compared per day. The number of videos has to be the double of the number of labels. In the rare case that this is not the case, there are 2 options:
<ul>
<li>Discard the day, especially if the difference is a big number of data.
<li>Check manually the label and the videos, it is usually quite rapid, ordering by name (by time), to identify the video with the label missing or vice-versa and make the correction manually.
</ul>
<li>The <code>DataPreparation.py</code> script is useful to chunk the video to the right size (if necessary) and then combine the two videos together. The script includes the option of stacking or concatenating the videos, as reported in the publication. This scripts creates a new folder containing the prepared dataset composed by the videos and a json file containing the labels.
<li>With the <code>DataSplitting.py</code> it is possible to reproduce the same splitting performed in the publication. The train.json, val.json and test.json are saved in a folder that should be named with the features chosen for the training.
<li> The training of the model can be done with <code>train.py</code>. Be sure of keeping track of the different hyperparameters chosen.
<li> The selected saved model can be evaluated using the <code>evaluate.py</code>, obtaining the classification metrics and the confusion matrices.
</ol>
