from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = RandomForestClassifier()
data = pd.read_csv('data/customerGroups.csv')
df = pd.DataFrame(data)
X = df.drop(['target', 'g1_21', 'g2_21', 'c_28'], axis=1).values
y = df['target'].values
# No need to divide into training and testing set
model.fit(X, y)


@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'prediction': 123})


@app.route("/predict", methods=["POST"])
def predict():
    features = request.json.get("features")
    feature_list = list(features)
    prediction = model.predict([feature_list])[0]
    prediction = int(prediction)
    return jsonify({"prediction": prediction})


if __name__ == '__main__':
    app.run(port=8000)
