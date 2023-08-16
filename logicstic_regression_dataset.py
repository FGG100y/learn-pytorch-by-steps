from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#  from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc # noqa


def moon_data():
    X, y = make_moons(n_samples=100, noise=0.3, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=13
    )

    sc = StandardScaler()
    sc.fit(X_train)

    X_train = sc.transform(X_train)
    X_val = sc.transform(X_val)

    return X_train, X_val, y_train, y_val
