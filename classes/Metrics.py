import numpy as np
class Metrics:
    def mean_absolute_error(y_test, y_pred):
        y_true, predictions = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs(y_true-predictions))
    
    def mean_squared_error(y_test, y_pred):
        y_true, predictions = np.array(y_true), np.array(y_pred)
        return np.mean((y_true-predictions)**2)
    
    def root_mean_squared_error(self,y_test, y_pred):
        return np.sqrt(self.mean_squared_error(y_test, y_pred))
    
    def mean_absolute_percentage_error(y_test, y_pred):
        y_true, predictions = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true-predictions)/y_true))
    
    def r_2_score(self, y_test, y_pred):
        y_true, predictions = np.array(y_true), np.array(y_pred)
        mean_value = np.mean(predictions)
        return self.mean_absolute_error(y_test, y_pred)/ np.mean((y_true-mean_value)**2)
    