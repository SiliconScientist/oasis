import numpy as np
from latent.model import get_residuals


def test_model(model_train, test_df):
    """
    Test a trained model on test data.

    Parameters:
    -----------
    model_train : trained model
        The trained model
    test_df : pd.DataFrame
        Test data
    Returns:
    --------
    x_test : pd.DataFrame
        Linearized test features
    y_test : pd.Series
        Test target values
    rmse : float
        Root mean square error
    mae : float
        Mean absolute error
    """
    train_params = model_train.train_params
    x_test, y_test = model_train.architecture.linearize_data(test_df, train_params)

    residuals = get_residuals(model_train, x_test, y_test)
    rmse = np.sqrt(np.mean((residuals) ** 2))
    mae = np.mean(np.abs((residuals)))

    return x_test, y_test, rmse, mae


def predict(model_train, x):
    """
    Make predictions using a trained model.

    Parameters:
    -----------
    model_train : trained model
        The trained model
    x : pd.DataFrame or array-like
        Features to predict on

    Returns:
    --------
    y_pred : np.array
        Predicted values
    """
    return model_train.predict(x)
