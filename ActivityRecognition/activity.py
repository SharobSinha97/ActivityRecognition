import os
import pickle as pkl

import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors

from ActivityRecognition import prepare_dataset


class ActivityModel(prepare_dataset.Dataset):

    def __init__(self, data_dir, model_dir):
        super(ActivityModel, self).__init__(data_dir=data_dir, model_dir=model_dir)
        self.model_file_name = os.path.join(model_dir, 'final_model.sav')
        self.data = self.prepare_dataset()
        self.X_train, self.X_test, self.Y_train, self.Y_test \
            = model_selection.train_test_split(self.data['features'], self.data['labels'], test_size=0.2,
                                               random_state=42)

    """
    setting visualise = true, leads to showing the model performance
    """

    def dump_model(self):
        model = neighbors.KNeighborsClassifier(n_neighbors=29, p=2)
        model.fit(self.X_train, self.Y_train)
        pkl.dump(model, open(self.model_file_name, 'wb'))

    def visualize_model(self):
        name = 'KNN'
        stats = []
        loaded_model = pkl.load(open(self.model_file_name, 'rb'))
        trainprediction = loaded_model.predict(self.X_train)
        testprediction = loaded_model.predict(self.X_test)
        scores = list()
        scores.append(name + "-train")
        scores.append(metrics.accuracy_score(self.Y_train, trainprediction))
        stats.append(scores)
        scores = list()
        scores.append(name + "-test")
        scores.append(metrics.accuracy_score(self.Y_test, testprediction))
        stats.append(scores)
        colnames = ["MODELNAME", "ACCURACY"]
        print(pd.DataFrame(stats, columns=colnames))
