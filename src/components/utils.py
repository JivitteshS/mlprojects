import os
import dill
import sys
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
from exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)




def evaluate_model(X_train,y_train,x_test,y_test,models,params):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)
            # model.fit(X_train,y_train).predict(X_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_prod =model.predict(X_train)

            y_test_prod=model.predict(x_test)

            train_model_score =r2_score(y_train,y_train_prod)

            test_model_score=r2_score(y_test,y_test_prod)

            report[list(models.keys())[i]]=test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
