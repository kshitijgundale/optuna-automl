import pandas as pd

class AutoML():
    
    def __init__(
        self,
        ml_task='auto',
        time_budget=3600,
        cv=5
    ): 
        self.ml_task = ml_task
        self.time_budget = time_budget
        self.cv = cv

        