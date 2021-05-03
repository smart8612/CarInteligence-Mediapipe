import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score
import pickle


modelFileName = "mymy.pkl"

if __name__ == '__main__':
    df = pd.read_csv('coords.csv')

    x = df.drop('class', axis=1)    # features: Pose, Face
    y = df['class']                 # target value

    # Partition Dataset for Train Model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

    # Configure Model Pipeline for inference
    pipelines = {
        'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    }

    # Train Model
    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(x_train, y_train)
        fit_models[algo] = model

    # Measure inferenced data Accuracy
    for algo, model in fit_models.items():
        inferenced_y = model.predict(x_test)
        print(algo, accuracy_score(y_test, inferenced_y))

    # Save model fitting result as binary data
    with open(modelFileName, 'wb') as f:
        pickle.dump(fit_models['lr'], f)

    exit()