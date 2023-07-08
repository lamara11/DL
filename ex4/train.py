import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
df = pd.read_csv('data.csv')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataset = ChallengeDataset(train_df, mode='train')
val_dataset = ChallengeDataset(val_df, mode='val')
train_dl = t.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dl = t.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


# create an instance of our ResNet model
resnet_model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
crit = t.nn.BCELoss()  # Binary Cross Entropy Loss for multi-label classification
# set up the optimizer (see t.optim)
optim = t.optim.Adam(resnet_model.parameters(), lr=0.001)
# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(
    model=resnet_model,
    crit=crit,
    optim=optim,
    train_dl=train_dl,
    val_test_dl=val_dl,
    early_stopping_patience=30
)

# go, go, go... call fit on trainer
res = trainer.fit(20)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')