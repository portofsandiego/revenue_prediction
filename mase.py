def mase(train, predict, actual, freq = None):
    """Computes Mean Absolute Scaled Error for a timeseries forecast.
    
    (Currently only works for seasonal timeseries)
    
    Args:
        train (list of floats): the training data used to generate the forecast.
        predict (list of floats): the forecasted values.
        actual (list of floats): real values to compare to predictions; must be same length as predict.
        freq (int, optional): the seasonal frequency of timeseries data used, if data is seasonal.
            Currently, must be specified. 
            
    Returns:
        float: the MASE value of the forecast against the test data.
    """
    
    T = len(train)
    qs = np.array([])
    
    if freq:
        
        for j in range(0, len(predict)):
            
            e = actual[j] - predict[j]
            
            naive_sum = 0
            for t in range(freq, T):
                naive_sum = naive_sum + abs(train[t] - train[t - freq])
                
            qs = np.append(qs, (e / ((1 / (T - freq)) * naive_sum)))
            
    return(np.abs(qs).mean())
