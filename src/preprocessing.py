from sklearn.preprocessing import StandardScaler

def preprocess_data(df):

    # Scale transaction amount
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # Time not useful initially
    df = df.drop("Time", axis=1)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y