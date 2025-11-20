

Hello, quick document to describe descisions made. 
will need to be refactored.

# Initial working NN 
At git commit: `60a8dce3073c4a276c082b58b9cf7bf2f73d5374`
Has batch sizes so we can train using mini batches which reduces overfitting.
Current working model using batch size of 1 so instead of mini batches we are training with SGD.
training data is not normalized and it is leading to NN getting stuck in local minimum when training.
examples of this can be seen here where the mean loss plateau's:
![image](50-epoch-batch-size-1-mean-loss.png)
![image](another-50-epoch-batch-size-1-mean-loss.png)


# new normalized training data
I believe this is because of the training data not being normalized between 0 and 1. to normalize the training data from -50 - 30, to between 0 and 1.
We will first normalize each angle in input and label data for training and testing. To do this we will get the difference between -50 and 30: which is 80. then we will + 50 to each angle and then / 80 to get the normalized result. For example angle -30 we will +50 so -30 + 50 = 20 then / 80 so 20/80 = 0.25.

once all the data is normalized and the NN is trained, we need to denormalize the predicted data. to do this we will take the normalized angle  * 80 and then -50 from it.

-50 and 30 are from the amplitude range of angles from target solution.

## results
