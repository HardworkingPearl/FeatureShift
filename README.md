<h1> Arousal Mental State Classification</h1>
This is a research project to do binary classification of EEG arousal mental state using deep learning
<h2> Dataset </h2>
The dataset is collected with Muse EEG headband and HTC Vive pro VR headset.
<lu>
  <li>Number of subjects: 18</li>
  <li>Classes: 2 (arousal vs calm)</li>
  <li>Sampling rate: 256 Hz</li>
  <li>Experiment protocol: (1 min calm + 1 min arousal + 5 s waitting) * 3</li>
  <li>Total duration for each subject: 6 mins</li>  
<h2>About the code </h2>
<p> Requirement: </p>
<ul>
  <li>pytorch == 1.2.0 or above</li>
  <li>h5py == 2.9.0</li>
  <li>scipy == 1.3.1</li>
  <li>numpy == 1.16.4</li>
</ul>

<h2>How to use</h2>
<p>Please use <code>Main.py</code> as the starting script </p>
<p>For training, use "[object of the class Train].set_parameter(<parameters>)" to modify the parameters. Please only use <object of the class Train>.Leave_one_session_out() to do subject-dependent classification task</p>
  <p> All the result will be save in to a txt file named <em>"result_leave_one_session_out_record.txt"</em> at the current directory of the script. Also, the Acc of training and evaluation can be found in <em>"Result_model/Leave_one_session_out/history/"</em> under the same directory.
    <h2>Paper Citation</h2>
    <p>If you find the code is useful, please cite our work:</p>
    <p><em>Author...,Decoding Emotional Arousal Mental State Using Frequency Correlation Graph Convolutional Neural Networks,Source</em></p>
    <p>Please try our best to get a high quality publication!</p>
    
    

