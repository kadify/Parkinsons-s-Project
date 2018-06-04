import pickle
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score



class gridsearch_models():
    '''
    Code to run:

    paramlist = [('GB', GradientBoostingClassifier(), {'n_estimators':[10, 50, 100, 200, 500],
                                                       'max_features':[None, 1, 2, 3, 4, 5, 6, 7, 8],
                                                       'max_depth':[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                                                       'learning_rate':[.01, .05, .1, 0.5, 1, 2],
                                                       'loss': ['deviance', 'exponential']})]

    gridsearch_models(df, 'status', sample_type='tomek', n=1000).gridsrch(paramlist)
    '''
    def __init__(self, df, target_column, sample_type=None, n=None):
        self.df = df
        self.target_column = target_column
        self.sample_type = sample_type
        self.n = n
        self._apply()


    def _apply(self):
        '''
        Cleans data, samples and bootstraps when specified and then returns output of models.
        '''
        self._preprocessing()
        self._split()

        if self.sample_type != None:
            self.sample()

        if self.n != None:
            self.bootstrapper()


    def _preprocessing(self):
        '''
        Returns dataframe with non numeric type columns removed
        '''
        toremove = []
        for column in self.df.columns:
            if self.df[column].dtypes != 'float64':
                 toremove.append(column)
        for i in toremove:
            if self.df[i].dtypes == 'int64':
                toremove.remove(i)
        for remove in toremove:
            self.df.drop([remove], axis=1, inplace=True)

    def sample(self):
        '''
        Dataframe generated with oversampled minority data or undersampled majority data depending on sample_type called.

        -------------------

        self.df = dataframe name you are trying to under or oversample

        self.target_column = column with 2 distinct values you wish to over or undersample. (Must be unbalanced)

        self.sample_type =
                        - "undersample" for undersampling of majority
                        - "oversample" for oversampling of minority
                        - "tomek" for SMOTETomek which oversamples the minority while removing tomek links
                            (majority class points are removed until the k nearest neighbors are the same class)

        -------------------
        '''
        cols = self.df.drop([self.target_column], axis=1).columns
        uniquevars = self.df[self.target_column].unique()

        if len(uniquevars) >2:
            return 'Error: there are more than 2 unique values in the given column: [{}]'.format('df', self.target_column)
        if len(self.df[self.df[self.target_column] == uniquevars[0]]) > len(self.df[self.df[self.target_column] == uniquevars[1]]):
            minority = (uniquevars[1])
            majority = (uniquevars[0])
        elif len(self.df[self.df[self.target_column] == uniquevars[0]]) < len(self.df[self.df[self.target_column] == uniquevars[1]]):
            minority = (uniquevars[0])
            majority = (uniquevars[1])
        else:
            return 'This dataframe is not imbalanced'


        df_majority = self.df[self.df[self.target_column]==majority]
        df_minority = self.df[self.df[self.target_column]==minority]
        if self.sample_type=='oversample':
            df_minority_oversample = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=12)
            df_sampled = pd.concat([df_majority, df_minority_oversample])
        if self.sample_type=='undersample':
            df_majority_undersample = resample(df_majority, replace=True, n_samples=len(df_minority), random_state=12)
            df_sampled = pd.concat([df_minority, df_majority_undersample])
        if self.sample_type=='tomek':
            x_resampled, y_resampled = SMOTETomek(random_state=42).fit_sample(self.df.drop([self.target_column], axis=1), self.df[self.target_column])
            df_sampled = pd.DataFrame(x_resampled, columns=cols)
            df_sampled[self.target_column] = y_resampled
        self.df = df_sampled


    def bootstrapper(self):
        '''
        If bootstrapping is specified, bootstraps to specified number of samples with replacement
        '''
        cols = self.df.columns
        dfnp = np.array(self.df)
        num_indeces = len(self.df)
        nrows = np.arange(0,num_indeces)
        new_df = np.random.choice(nrows, self.n, replace=True)
        new_df = pd.DataFrame(dfnp[new_df,:])
        new_df.columns = cols
        self.df = new_df

    def _split(self):
        '''
        Splits data into training dataframe and x and y test arrays. Test arrays used to verify generalizability of model
        '''
        x_train, x_test, y_train, y_test =  train_test_split(self.df.drop([self.target_column],axis=1), self.df[self.target_column], test_size=.45, stratify=self.df[self.target_column], random_state=12)
        scaler = StandardScaler().fit(x_train, x_test)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        self.x_train, self.x_test = x_train, x_test
        self.y_train, self.y_test = y_train, y_test

        cols = self.df.drop([self.target_column], axis=1).columns

        df = pd.DataFrame(x_train, columns=cols)
        y = pd.DataFrame(y_train, dtype='int64')

        df = df.join(y)
        self.df = df.fillna(0)
        self.df[self.target_column] = self.df[self.target_column].astype(int)


    def _fit(self, model_best):
        '''
        Fits training data on best model from gridsrch, predicts precision, recall, accuracy and auc and
        returns feature importances, precision, recall and accuracy of model, confusion matrix of predictions,
        and ROC curve
        '''

        final = []
        model = model_best
        model.fit(self.x_train, self.y_train)
        self.y_predict_model = model.predict(self.x_test)
        final_precision = precision_score(self.y_test, self.y_predict_model)
        final_recall = recall_score(self.y_test, self.y_predict_model)
        final_accuracy = accuracy_score(self.y_test, self.y_predict_model)

        final.append((self.name, 'Precision: '+ str(format(final_precision, '.3f')), 'Recall: '+ str(format(final_recall, '.3f'))+'\t', 'Accuracy: ' + str(format(final_accuracy, '.3f'))+'\t'))

        feature_importances = pd.DataFrame(model.feature_importances_, index=self.df.drop([self.target_column],axis=1).columns, columns=['importance']).sort_values('importance', ascending=False)
        print(feature_importances)
        print()
        for i in final[0]:
            print(i)
#             for line in zip(final[0], val[0]):
#                 print('{0}\t\t\t{1}'.format(*line))
        self.standard_confusion_matrix()
        self.ROCYOU(model)

        print('='*60,'\n')



    def standard_confusion_matrix(self, standardm=False):
        '''
        Prints confusion matrix and guide matrix of predictions
        '''
        a = (confusion_matrix(self.y_test, self.y_predict_model))
        a = np.fliplr(a).T
        b = np.fliplr(a).T
        standard = np.array([['TP', 'FN'],['FP', 'TN']])

        print((tabulate(standard, tablefmt="grid")))

        print((tabulate(np.array(b), tablefmt="grid")))


    def gridsrch(self, param_pipeline_list, score='f1'):
        '''
        Returns and pickles best fit model for data based on list of models provided.

        Expects list with name, model type, and parameters dictionary.
            Ex. [('RF', RandomForestClassifier(), {'n_estimators':[10, 50, 100, 200, 500], etc.})]

        score = scoring metric, defaults to f1. Potential other metrics include roc_auc, Accuracy, etc.
        '''
        pipelines = param_pipeline_list


#     [('RF', RandomForestClassifier(), {'n_estimators':[10, 50, 100, 200, 500],
#                                                    'max_features':[None, 1, 2, 3, 4, 5, 6, 7, 8],
#                                                    'min_samples_split': [2,5,10,20],
#                                                    'max_depth':[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
#                                                   'bootstrap': [False],
#                                                   'class_weight':[None, 'balanced_subsample', 'balanced']}),
#                  ('GB', GradientBoostingClassifier(), {'n_estimators':[10, 50, 100, 200, 500],
#                                                        'max_features':[None, 1, 2, 3, 4, 5, 6, 7, 8],
#                                                        'max_depth':[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
#                                                        'learning_rate':[.01, .05, .1, 0.5, 1, 2],
#                                                        'loss': ['deviance', 'exponential']})]


        for name, model, param in pipelines:
            model_grid = GridSearchCV(model, param, scoring=score, n_jobs=-1, refit=True)
            print('Fitting ' + name)
            model_grid.fit(self.x_train, self.y_train)
            model_best = model_grid.best_params_
            print('Pickling ' + name)
            print(model_best)
            model_best = model_grid.best_estimator_
            pickle.dump(model_grid, open(name+'.pkl', 'wb'))
            pickle.dump(model_best, open(name+'bestpars.pkl', 'wb'))
            self.name = name
        self._fit(model_best)

    def ROCYOU(self, model):
        '''
        Outputs ROC curve based on
        '''
        fpr, tpr, _ = roc_curve(self.y_test,  self.y_predict_model)
        self.auc = roc_auc_score(self.y_test,  self.y_predict_model)
        plt.plot(fpr,tpr,label="data 1, auc="+str(self.auc))
        plt.legend(loc=4)
        plt.show()
