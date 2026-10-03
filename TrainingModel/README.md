# Data preparation and model training

The system records two videos for each label. Before training, check and prepare the data by following these steps in order:

<ol>
<li>Use <code>VideoLabelCounter.py</code> to compare the number of labels and videos for each day. There should be twice as many videos as labels. If the counts do not match, you can either:
<ul>
<li>Discard that day's data, especially if the counts differ substantially.
<li>Manually inspect the labels and videos. Sort them by name (and therefore by time) to identify and correct any missing label or video.
</ul>
<li>Use <code>DataPreparation.py</code> to divide videos into clips of the required length, if necessary, and combine the two camera videos. The script can stack or concatenate the videos, as described in the publication. It creates a folder containing the prepared videos and a JSON file with their labels.
<li>Use <code>DataSplitting.py</code> to reproduce the data split used in the publication. The script saves <code>train.json</code>, <code>val.json</code>, and <code>test.json</code> in a folder named for the features selected for training.
<li>Train the model with <code>train.py</code>. Keep track of the hyperparameters used.
<li>Evaluate the saved model with <code>evaluate.py</code> to obtain classification metrics and confusion matrices.
</ol>
