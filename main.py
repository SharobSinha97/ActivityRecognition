from ActivityRecognition import activity

if __name__ == '__main__':
    data_dir = './data/'
    model_dir = './models'

    activity_recognizer = activity.ActivityModel(data_dir=data_dir, model_dir=model_dir)
    activity_recognizer.dump_model()
    activity_recognizer.visualize_model()
