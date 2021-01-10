import os
import pandas as pd


class Dataset:
    def __init__(self, data_dir, model_dir):
        self.data_dir = data_dir
        self.model_dir = model_dir

    def prepare_dataset(self):
        li = []
        for file_name in os.listdir(self.data_dir):
            current_df = pd.read_csv(os.path.join(self.data_dir, file_name), header=None)
            current_df[0] = file_name
            li.append(current_df)
        dataframe = pd.concat(li)
        dataframe = dataframe[dataframe[4] != 0]
        dataframe = dataframe[dataframe[4] != 2]
        dataframe = dataframe[dataframe[4] != 5]
        dataframe = dataframe[dataframe[4] != 6]
        dataframe.rename(columns={0: 'filename', 1: 'x_acc', 2: 'y_acc', 3: 'z_acc', 4: 'label'}, inplace=True)
        # write df as a complete dataset

        dataframe.to_csv(os.path.join(self.model_dir, 'dataset.csv'))
        return {
            "features": dataframe.drop(['filename', 'label'], axis=1),
            "labels": dataframe['label']
        }
        # Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=.2, random_state=42)
        # write Xtrain, Xtest, test, train in a csv file
