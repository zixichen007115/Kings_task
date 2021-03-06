# Kings_task
## task1
### description: 
In most real-world applications, labelled data is scarce. Suppose you are given the MNIST dataset, but without any labels in the training set. The labels are held in a database,
which you may query to reveal the label of any particular image it contains. Your task is to build a classifier
to >90% accuracy on the test set, using the smallest number of queries to this database.

You may use any combination of techniques you find suitable (supervised, self-supervised, unsupervised).
However, using other datasets or pre-trained models is not allowed.

### strategy:
choose typical and various samples from dataset

extract features from images - select images according to their features - train a classifier

### method:
1.(baseline) random selection + CNN 500 images

2.PCA + KMEANS + CNN

3.AE + KMEANS +CNN

4.submodular active learning[1]

![diagram](https://github.com/zixichen007115/Kings_task/blob/main/diagram.png "diagram")

### result:
1.acc

![test_acc](https://github.com/zixichen007115/Kings_task/blob/main/method_acc.png "the acccurancy of every method")

2.Kmeans parameter influence

![cluster_acc](https://github.com/zixichen007115/Kings_task/blob/main/cluster_acc.png "the acccurancy of every cluster number")


3.best method

minimal number:350-400

![best](https://github.com/zixichen007115/Kings_task/blob/main/best.png "best method")

## task2
### description: 
Re-implement in Python the results presented in Example 6.6 of the Sutton & Barto book on page 132
comparing SARSA and Q-learning in the cliff-walking task. Investigate the effect of choosing different values
for the exploration parameter ε for both methods. Present your code and results. In your discussion clearly
describe the main difference between SARSA and Q-learning in relation to your findings.
Note: For this problem, use 𝛼 = 0.1 and 𝛾 = 1 for both algorithms. The "smoothing" that is mentioned in the
caption of Figure 6.4 is a result of 1) averaging over 10 runs, and 2) plotting a moving average over the last
10 episodes.

### result:

![map](https://github.com/zixichen007115/Kings_task/blob/main/map.jpg "map")
![reward](https://github.com/zixichen007115/Kings_task/blob/main/reward.png "reward")
![reward_de](https://github.com/zixichen007115/Kings_task/blob/main/reward_de.png "reward_de")

### discussion:
Compared with SARAS, Q-learning uses the maximum of the following q, hence it is more positive and will ignore penalty caused by randomly stepping into the cliff.
SARAS calcuates q by deciding actions first, which means it will consider all available future states, so state near the cliff is more dangerous.

SARAS is on-policy since it will explore the state and update q, while Q-learning only update q based on greedy policy before deciding actions.

## Bibliography
[1] Kaushal V, Sahoo A, Doctor K, et al. Learning from less data: Diversified subset selection and active learning in image classification tasks[J]. arXiv preprint arXiv:1805.11191, 2018.
