# uni_D83AE5_uni (Baseline)
#   {"Logloss": 0.024257891067585, "ROCAUC": 0.9824559651042923}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats import gamma, kstest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    log_loss, 
    roc_auc_score, 
    roc_curve, 
    auc
)
import statsmodels.api as sm
import warnings

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

warnings.filterwarnings('ignore')  # Use this line to suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def gen(params={
    'RUN_NAME': 'default',
    'LR_R': 1.58,
    'RF_R': 2.08,
    'RF_N': 600,
    'RF_D': 11,
    'RF_L': 100,
    'FEATURES':0.005,
}):

    training_data = pd.read_csv('training_data.csv')
    training_data = training_data[training_data['BORROWER_ID'] != 'xNullx']
    training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)
    data_submission_example = pd.read_csv('data_submission_example.csv')

    lognormal_variables = [
        'CONTRACT_CREDIT_LOSS', 'CONTRACT_DEPT_SERVICE_TO_INCOME',
        'CONTRACT_INCOME', 'CONTRACT_INSTALMENT_AMOUNT', 'CONTRACT_INSTALMENT_AMOUNT_2',
        'CONTRACT_LOAN_AMOUNT', 'CONTRACT_MARKET_VALUE', 'CONTRACT_MORTGAGE_LENDING_VALUE', 
        'CONTRACT_LGD', 'CONTRACT_INCOME'
    ]

    training_data.fillna(0, inplace=True)
    for col in training_data.columns:
        try:
            training_data[col] = pd.to_numeric(training_data[col], errors='ignore')
        except:
            continue
    training_data['TARGET_EVENT_BINARY'] = np.where(training_data['TARGET_EVENT'] == 'K', 1, 0)


    training_data['TARGET_EVENT_E'] = np.where(training_data['TARGET_EVENT'] == 'E', 1, 0)


    date_variables = ['CONTRACT_DATE_OF_LOAN_AGREEMENT', 'CONTRACT_MATURITY_DATE']
    training_data['TARGET_EVENT_DAY'].replace(0.0, np.nan, inplace=True)
    training_data['TARGET_EVENT_DAY_JULIAN'] = pd.to_datetime(training_data['TARGET_EVENT_DAY'], origin='julian', unit='D', errors='coerce')
    training_data['TARGET_EVENT_DAY_DATETIME'] = pd.to_datetime(training_data['TARGET_EVENT_DAY_JULIAN'],  errors='coerce')

    training_data['CONTRACT_DATE_OF_LOAN_AGREEMENT_JULIAN'] = pd.to_datetime(training_data['CONTRACT_DATE_OF_LOAN_AGREEMENT'], origin='julian', unit='D')
    training_data['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME'] = pd.to_datetime(training_data['CONTRACT_DATE_OF_LOAN_AGREEMENT_JULIAN'],)

    training_data['CONTRACT_MATURITY_DATE_JULIAN'] = pd.to_datetime(training_data['CONTRACT_MATURITY_DATE'], origin='julian', unit='D')
    training_data['CONTRACT_MATURITY_DATE_DATETIME'] = pd.to_datetime(training_data['CONTRACT_MATURITY_DATE_JULIAN'])
    training_data['DAY_DIFF'] = (training_data['TARGET_EVENT_DAY_DATETIME'] - training_data['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME']).dt.days
    training_data['DAYS_TO_END'] = (pd.Timestamp("2020-01-01")- training_data['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME']).dt.days
    training_data['YEARS_TO_END'] = training_data['DAYS_TO_END'] / 365
    training_data['DAYS_TO_2018'] = (pd.Timestamp("2018-01-01")- training_data['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME']).dt.days
    training_data['YEARS_TO_2018'] = training_data['DAYS_TO_2018'] / 365
    training_data['TIME_TO_MATURITY_DAYS'] = (training_data['CONTRACT_MATURITY_DATE']-training_data['CONTRACT_DATE_OF_LOAN_AGREEMENT'])
    training_data['2020_OR_MATURITY'] = np.minimum(training_data['TIME_TO_MATURITY_DAYS'], training_data['DAYS_TO_END'])
    training_data['2020_OR_MATURITY_YEARS'] = training_data['2020_OR_MATURITY'] / 365

    def create_binary_target_column(dataframe, column_name, event, day_diff_upper_limit):
        dataframe[column_name] = np.where(
            (dataframe['TARGET_EVENT'] == event) & 
            (dataframe['DAY_DIFF'] <= day_diff_upper_limit) & 
            (dataframe['DAY_DIFF'] >= 0), 
            1, 
            0
        )

    timeframes = {
        'TARGET_EVENT_BINARY_2Y': 730,
        'TARGET_EVENT_BINARY_1Y': 365,
        'TARGET_EVENT_BINARY_6M': 365//2,
    }

    for column_name, days in timeframes.items():
        create_binary_target_column(training_data, column_name, 'K', days)


    training_data.drop('DAY_DIFF', axis=1, inplace=True)
    print(training_data['TARGET_EVENT_BINARY'].value_counts())
    print(training_data['TARGET_EVENT_BINARY_2Y'].value_counts())
    print(training_data['TARGET_EVENT_BINARY_1Y'].value_counts())
    print(training_data['TARGET_EVENT_BINARY_6M'].value_counts())


    numeric_columns = training_data.select_dtypes(include=[np.number]).columns.tolist()
    excluded_keywords = ['TARGET', 'event', 'binary', 'DATE', 'DAYS', 'YEARS', 'MATURITY', 'DAY']

    X_columns = [col for col in numeric_columns if all(keyword.lower() not in col.lower() for keyword in excluded_keywords)]
    y_column = 'TARGET_EVENT_BINARY_2Y' 

    loan_type_dummies = pd.get_dummies(training_data['CONTRACT_LOAN_TYPE'], prefix='LOAN_TYPE', drop_first=True)
    frequency_type_dummies = pd.get_dummies(training_data['CONTRACT_FREQUENCY_TYPE'], prefix='FREQ_TYPE', drop_first=True)
    interest_type_dummies = pd.get_dummies(training_data['CONTRACT_TYPE_OF_INTEREST_REPAYMENT'], prefix='INTEREST_TYPE', drop_first=True)
    mortgage_type_dummies =  pd.get_dummies(training_data['CONTRACT_MORTGAGE_TYPE'], prefix='MORTGAGE_TYPE', drop_first=True)

    training_data = pd.concat([training_data, loan_type_dummies, frequency_type_dummies,interest_type_dummies,mortgage_type_dummies ], axis=1)

    X_columns.extend(loan_type_dummies.columns)
    X_columns.extend(frequency_type_dummies.columns)
    X_columns.extend(interest_type_dummies.columns)
    X_columns.extend(mortgage_type_dummies.columns)
    # X_columns.remove('CONTRACT_LOAN_TYPE')
    # X_columns.remove('CONTRACT_FREQUENCY_TYPE')
    X_columns.remove('CONTRACT_TYPE_OF_INTEREST_REPAYMENT')
    X_columns.remove('CONTRACT_MORTGAGE_TYPE')


    training_data['BORROWER_LOAN_COUNT'] = training_data.groupby('BORROWER_ID')['BORROWER_ID'].transform('count')
    training_data['LOAN_BORROWER_COUNT'] = training_data.groupby('CONTRACT_ID')['CONTRACT_ID'].transform('count')
    training_data['TOTAL_LOAN_AMOUNT'] = training_data.groupby('BORROWER_ID')['CONTRACT_LOAN_AMOUNT'].transform('sum')
    training_data['TOTAL_INSTALLMENT_AMOUNT_1'] = training_data.groupby('BORROWER_ID')['CONTRACT_INSTALMENT_AMOUNT'].transform('sum')
    training_data['TOTAL_INSTALLMENT_AMOUNT_2'] = training_data.groupby('BORROWER_ID')['CONTRACT_INSTALMENT_AMOUNT_2'].transform('sum')
    training_data['TOTAL_INSTALLMENT_AMOUNT'] = training_data['TOTAL_INSTALLMENT_AMOUNT_1'] + training_data['TOTAL_INSTALLMENT_AMOUNT_2']

    X_columns.extend(['BORROWER_LOAN_COUNT', 'TOTAL_LOAN_AMOUNT','TOTAL_INSTALLMENT_AMOUNT','LOAN_BORROWER_COUNT'])
    lognormal_variables.extend([ 'TOTAL_LOAN_AMOUNT','TOTAL_INSTALLMENT_AMOUNT'])

    threshold = 0.85
    correlation_matrix = training_data[X_columns].corr()
    highly_correlated_set = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                highly_correlated_set.add(colname)

    X_columns = [col for col in X_columns if col not in highly_correlated_set]
    print('Variables removed:', highly_correlated_set)

    def default_percentage_generator_2016(df, percentage, target, random_sample=42):
        df_copy = df.copy()
        df_filtered = df_copy[df_copy['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME'] < '2016-01-01']
        df_mean = df_filtered[target].mean()
        print(f"Mean in year {df_mean}")
        df_defautled = df_filtered[target].sum()
        df_not_defaulted = len(df_filtered) - df_defautled

        required_val = (df_defautled - percentage * len(df_filtered))/(percentage - 1)
        
        df_filtered_after = df_copy[df_copy['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME'] > '2016-01-01']
        df_filtered_after = df_filtered_after[df_filtered_after[target]==1] 
        print(len(df_filtered_after))
        print(required_val)
        required_val = min(int(required_val) ,len(df_filtered_after))
        df_filtered_after = df_filtered_after.sample(n=int(required_val),random_state=random_sample).reset_index(drop=True)

        df_filtered = pd.concat([df_filtered, df_filtered_after])

        return df_filtered

    def default_percentage_generator_2016_maximal(df, percentage, target):
        df_copy = df.copy()

        defaulted = df_copy[(df_copy[target]==1) & (df_copy['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME'] < '2018-01-01')]
        total_needed = len(defaulted) / percentage * 100 - len(defaulted)


        # print(len(defaulted))
        # print(total_needed)

        df_filtered_2016 = df_copy[(df_copy['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME'] < '2016-01-01')& (df_copy[target]==0)]
        # print(len(df_filtered_2016))
        # print(total_needed)
        max_needed = min(total_needed,len(df_filtered_2016))
        df_filtered_2016 = df_filtered_2016.sample(n=int(max_needed),random_state=42).reset_index(drop=True)

        extra_needed = total_needed - len(df_filtered_2016)

        df_filtered_after = df_copy[df_copy['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME'] > '2016-01-01']
        df_filtered_after = df_filtered_after[df_filtered_after[target]==0]
        df_filtered_after = df_filtered_after.sample(n=int(extra_needed),random_state=42).reset_index(drop=True)


        df_filtered = pd.concat([df_filtered_2016, df_filtered_after,defaulted])

        return df_filtered

    # default_percentage_generator_2016_maximal(training_data, 1.48, 'TARGET_EVENT_BINARY')['TARGET_EVENT_BINARY'].mean()
    def calculate_probabilities(data, column, time_factor):
        lambdas = -np.log(1 - data[column]) / time_factor
        probs_2y = 1 - np.exp(-2 * lambdas)
        return probs_2y
    def calculate_probabilities_vec(data, time_factor):
        lambdas = -np.log(1 - data) / time_factor
        probs_2y = 1 - np.exp(-2 * lambdas)
        return probs_2y

    from sklearn.calibration import CalibratedClassifierCV
    from imblearn.over_sampling import SMOTE
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import brier_score_loss




    def train_and_predict_two_halves(df, variables, target, model=LogisticRegression(), 
                                    scaler=StandardScaler(), augment_distribution=True,calibrate=True,
                                    augment_distribution_percentage = 1.48, unique_loans=False,
                                    should_smote =False,maximal_sample=False,random_sample=42, 
                                    show_curve=False, calib_method = 'isotonic', cv_validate=False):
        if lognormal_variables is not None:
            df = df.copy()
            
            for var in lognormal_variables:
                if var == 'CONTRACT_CREDIT_LOSS':
                    df[var] = np.log1p(np.abs(df[var]))*np.sign(df[var])
                else:
                    df[var] = np.log1p(df[var])
        if augment_distribution and not maximal_sample:
            df_filtered = default_percentage_generator_2016(df, augment_distribution_percentage/100, target, random_sample)
        else:
            df_filtered = df[df['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME'] < '2016-01-01']
        
        if maximal_sample:
            df_filtered = default_percentage_generator_2016_maximal(df, augment_distribution_percentage, target)
            print(df_filtered[target].mean())

        if unique_loans:
            df_filtered = df_filtered.drop_duplicates(subset=['CONTRACT_ID'])

        
        X_scaled = scaler.fit_transform(df[variables])
        X_filtered = scaler.transform(df_filtered[variables])

        y = df[target] 
        y_filtered = df_filtered[target] 
        if should_smote:
            smote = SMOTE(random_state=42)
            X_filtered, y_filtered = smote.fit_resample(X_filtered, y_filtered)

        if calibrate:
            model = CalibratedClassifierCV(base_estimator=model, method=calib_method, )

        
        cv_predictions_filtered = cross_val_predict(model, X_filtered, y_filtered, cv=5, method='predict_proba')
        df_filtered['cv_pred'] = cv_predictions_filtered[:, 1]  # Assuming binary classification

        model.fit(X_filtered, y_filtered)


        print("Logloss:")
        test_proba = model.predict_proba(X_filtered)
        print(log_loss(y_filtered,test_proba))
        print("Logloss CV (Should be main metric):")
        print(log_loss(y_filtered,cv_predictions_filtered))
        brier_score = brier_score_loss(y_filtered, cv_predictions_filtered[:,1])
        print(f"Brier Score: {brier_score}")
        proba = model.predict_proba(X_scaled)[:, 1]
        df['model_pred'] = proba
        print('Model mean on all data:', df['model_pred'].mean())

        combined_df = df.merge(df_filtered[['CONTRACT_ID', 'BORROWER_ID', 'cv_pred']], 
                            on=['CONTRACT_ID', 'BORROWER_ID'], 
                            how='left')

        # Replace predictions in df with those from cross-validation where available
        combined_df.loc[combined_df.cv_pred.notna(), 'model_pred'] = combined_df.cv_pred
        print('Moddel mean on CV combined:', combined_df['model_pred'].mean())
        roc_auc = roc_auc_score(y_filtered,test_proba[:,1])
        print("ROC AUC Score:", roc_auc)
        roc_auc = roc_auc_score(y_filtered,cv_predictions_filtered[:,1])
        print("ROC AUC Score CV (should be main metric):", roc_auc)

        if show_curve:
            true_probas, predicted_probas = calibration_curve(y_filtered, test_proba[:, 1], n_bins=10)
            plt.figure(figsize=(8, 6))
            plt.plot(predicted_probas, true_probas, marker='o')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Perfect calibration line
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Curve')
            plt.show()
        
        
        return combined_df['model_pred'], model

    def significant_features(df, variables, target, model1=LogisticRegression(), scaler=StandardScaler()):
        if lognormal_variables is not None:
            df = df.copy()
            print(df[variables].isna().sum().sum())
            
            for var in lognormal_variables:
                if var == 'CONTRACT_CREDIT_LOSS':
                    df[var] = np.log1p(np.abs(df[var]))*np.sign(df[var])
                else:
                    df[var] = np.log1p(df[var])
        print(np.isinf(df[variables]).sum().sum())
        X_scaled = scaler.fit_transform(df[variables])
        print(np.isinf(X_scaled).sum().sum())
        print(pd.DataFrame(X_scaled, columns=variables).isna().sum().sum())
        X = sm.add_constant(pd.DataFrame(X_scaled, columns=variables))
        y = df[target]
        model = sm.Logit(y, X).fit()
        print(model.summary())
        p_values = model.pvalues
        non_significant_vars = p_values[p_values > 0.05].index.tolist()

        return model, non_significant_vars


    def combined_probability(s):
        if len(s) == 2:
            p_a, p_b = s.values
            return p_a + p_b - p_a * p_b #- (-3.2357827075016176e-05)
        else:
            return 1 - np.prod(1 - s.values)

    def create_submission_file(df_preds, target, example, filename='submission.csv', testing=False):
        # Filter the data to only include BORROWER_IDs that are in the submission example
        df_preds.loc[df_preds['TARGET_EVENT'] == 'E', target] = 0

        print("Logloss:")
        print(log_loss(df_preds['TARGET_EVENT_BINARY'], df_preds[target]))

        filtered_training_data = df_preds[df_preds['BORROWER_ID'].isin(example['BORROWER_ID'])]

        # Print warning if the row count is off
        if not testing and len(filtered_training_data) != 1564601:
            print('WARNING: The filtered data does not have the correct number of rows. Make sure you are not using the training data for submission.')
            
        grouped_data = filtered_training_data.groupby('BORROWER_ID')[target].apply(combined_probability).reset_index()

        # Create the submission DataFrame
        df_submission = pd.DataFrame()
        df_submission['BORROWER_ID'] = grouped_data['BORROWER_ID']
        df_submission['PRED'] = grouped_data[target]
        print('Before centering:')
        print(df_submission['PRED'].max(), df_submission['PRED'].min(), df_submission['PRED'].mean())

        print('Centering probabilities...')
        # Center the probabilities around 1.48%
        desired_mean = 0.0148  
        initial_guess = 2
        probas_unscaled = df_submission['PRED'].values
        new_proba = probas_unscaled.copy()
        while abs(new_proba.mean() - desired_mean) > 0.00001:
            
            new_proba = calculate_probabilities_vec(probas_unscaled, initial_guess)
            error = new_proba.mean() - desired_mean
            if error > 0:
                initial_guess += 0.001
            else:
                initial_guess -= 0.001
            print(error, initial_guess)
        df_submission['PRED'] = new_proba
            
            
        
        print(df_submission['PRED'].max(), df_submission['PRED'].min(), df_submission['PRED'].mean())
        # Save the submission file
        if  not testing and filename is not None:
            df_submission.to_csv(filename, index=False)
        print(f'Saved file: {filename}')
        # if abs(df_submission['PRED'].mean() -0.0148) > 0.0005:
        #    raise ValueError('WARNING: mean is bad')
            
        # Print warning if the row count is off
        if not testing and len(df_submission) != 1117674:
            print('WARNING: The submission file does not have the correct number of rows. Make sure you are not using the training data for submission.')
            # raise ValueError('WARNING: The submission file does not have the correct number of rows. Make sure you are not using the training data for submission.')
            
        return df_submission

    def scale_yearly_proba(data, proba, targets =[ 0.0052, 0.0099, 0.0185], logging=False):
        data = data.copy()
        probs = data[proba]
        starter_scales=[2.8, 1.2, 0.75]
        new_proba = np.zeros(len(data))
        mask_2016 = (data['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME'] < '2016-01-01') & (data['TARGET_EVENT_BINARY'] != 1)
        mask_2017 = (data['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME'] > '2016-01-01') & (data['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME'] < '2017-01-01')& (data['TARGET_EVENT_BINARY'] != 1)
        mask_2018 = (data['CONTRACT_DATE_OF_LOAN_AGREEMENT_DATETIME'] > '2017-01-01') & (data['TARGET_EVENT_BINARY'] != 1)

        proba_2016 = probs
        proba_2017 = probs
        proba_2018 = probs
        if logging:
            print("Before scaling:")
            print(proba_2016[mask_2016].mean(), proba_2017[mask_2017].mean(), proba_2018[mask_2018].mean())
            print(probs.mean())
        calib_2016 = True
        calib_2017 = True
        calib_2018 = True

        while calib_2016 or calib_2017 or calib_2018:    
            proba_2016 = probs
            proba_2017 = probs
            proba_2018 = probs
            proba_2016 = calculate_probabilities_vec(proba_2016, starter_scales[0])
            proba_2017 = calculate_probabilities_vec(proba_2017, starter_scales[1])
            proba_2018 = calculate_probabilities_vec(proba_2018, starter_scales[2])
            
            new_proba[mask_2016] = proba_2016[mask_2016]
            new_proba[mask_2017] = proba_2017[mask_2017]
            new_proba[mask_2018] = proba_2018[mask_2018]
            if logging:
                print('Adter scaling:')
                print(proba_2016[mask_2016].mean(), proba_2017[mask_2017].mean(), proba_2018[mask_2018].mean())
                print(new_proba.mean())

            mean_2016 = proba_2016[mask_2016].mean()
            mean_2017 = proba_2017[mask_2017].mean()
            mean_2018 = proba_2018[mask_2018].mean()

            diff_2016 = mean_2016 - targets[0]
            diff_2017 = mean_2017 - targets[1]
            diff_2018 = mean_2018 - targets[2]
            if diff_2016 > 0.0001:
                starter_scales[0] += 0.01
            elif diff_2016 < -0.0001:
                starter_scales[0] -= 0.01
            else:
                calib_2016 = False
            if diff_2017 > 0.0001:
                starter_scales[1] += 0.01
            elif diff_2017 < -0.0001:
                starter_scales[1] -= 0.01
            else:
                calib_2017 = False
            if diff_2018 > 0.0001:
                starter_scales[2] += 0.01
            elif diff_2018 < -0.0001:
                starter_scales[2] -= 0.01
            else:
                calib_2018 = False
        return new_proba
        

    X_columns.remove('CONTRACT_CURRENCY')
    X_columns.remove('BORROWER_COUNTY')
    # X_columns.remove('CONTRACT_MORTGAGE_TYPE')
    X_columns.remove('BORROWER_TYPE_OF_SETTLEMENT')
    X_columns.remove('BORROWER_CITIZENSHIP')
    X_columns.remove('LOAN_BORROWER_COUNT')


    from sklearn.feature_selection import VarianceThreshold

    def variance_threshold_selector(data, threshold=0.5):
        # https://stackoverflow.com/a/39813304/1956309
        selector = VarianceThreshold(threshold)
        selector.fit(data)
        return data.columns[selector.get_support(indices=True)]
    min_variance = 0.0001
    low_variance = variance_threshold_selector(training_data[X_columns], min_variance) 

    for i in X_columns:
        if i not in low_variance:
            print(i)

    X_columns = low_variance
    X_columns= list(X_columns)

    vals = training_data['CONTRACT_LOAN_TYPE'].value_counts() < 5000


    small_list = vals.index[vals].tolist()
    small_header = ['LOAN_TYPE_' + i for i in small_list]
    print(small_header)
    for i in small_header:
        if i in X_columns:
            print(i)
            X_columns.remove(i)


    model, non_significant_vars = significant_features(training_data, X_columns, y_column,LogisticRegression())

    predicted_probs = 'XGB'
    training_data['XGB_CUMM'] = 0
    iterations = 1
    for i in range(iterations):
        probs, xgb_model = train_and_predict_two_halves(
            training_data, 
            X_columns, 
            'TARGET_EVENT_BINARY',
            model=xgb.XGBClassifier(max_depth=2, n_estimators=70, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            augment_distribution=True,
            augment_distribution_percentage=1.68,
            calibrate=False,
            random_sample=i,
            calib_method='isotonic',
            show_curve=False,
        )
        training_data[predicted_probs] = probs
        training_data['XGB_CUMM'] += probs
    training_data['XGB_CUMM'] /= iterations
    print(probs.mean())
    # this helps
    training_data.loc[training_data['TARGET_EVENT'] == 'E', 'XGB'] = 0
    training_data.loc[training_data['TARGET_EVENT'] == 'E', 'XGB_CUMM'] = 0
    feature_importances = xgb_model.feature_importances_
    feature_names = X_columns

    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_feature_importances = feature_importances[sorted_idx]
    sorted_feature_names = np.array(feature_names)[sorted_idx]

    # Plot
    # plt.figure(figsize=(10, 10))
    # sns.barplot(x=sorted_feature_importances, y=sorted_feature_names, palette="viridis")
    # plt.title('Sorted Feature Importances')
    # plt.show()


    zero_importance_features = np.array(feature_names)[feature_importances < params['FEATURES']]
    zero_importance_features = list(zero_importance_features)
    XGB_features = X_columns.copy()
    for i in zero_importance_features:
        if i in XGB_features:
            print(i)
            XGB_features.remove(i)
    print(len(XGB_features))
    predicted_probs = 'LOGISTIC_REG'
    training_data['LOGISTIC_REG_CUMM'] = 0
    iterations = 10
    for i in range(iterations):
        probs,_ = train_and_predict_two_halves(
            training_data, 
            XGB_features, 
            'TARGET_EVENT_BINARY',
            model=LogisticRegression(max_iter=400, random_state=42,solver='lbfgs'),
            augment_distribution=True,
            augment_distribution_percentage=params['LR_R'],
            calibrate=True,
            unique_loans=False,
            should_smote=False,
            maximal_sample=False,
            random_sample=i,
            calib_method='isotonic',
            show_curve=(i==0),
            cv_validate=True,
        )
        training_data[predicted_probs] = probs
        training_data['LOGISTIC_REG_CUMM'] += probs
    training_data['LOGISTIC_REG_CUMM'] /= iterations
    print(probs.mean())
    training_data.loc[training_data['TARGET_EVENT'] == 'E', 'LOGISTIC_REG'] = 0
    training_data.loc[training_data['TARGET_EVENT'] == 'E', 'LOGISTIC_REG_CUMM'] = 0



    predicted_probs = 'NN'
    training_data['RF_CUMM'] = 0
    iterations = 2
    for i in range(iterations):
        probs,_ = train_and_predict_two_halves(
            training_data, 
            XGB_features, 
            'TARGET_EVENT_BINARY',
            model= RandomForestClassifier(n_estimators=params['RF_N'], max_depth=params['RF_D'], random_state=42, min_samples_leaf=params['RF_L'], criterion="log_loss", n_jobs=-1   ),
            augment_distribution=True,
            augment_distribution_percentage=params['RF_R'],
            calibrate=True,
            random_sample=i,
            calib_method='isotonic',
            show_curve=(i==0),
        )
        training_data[predicted_probs] = probs
        training_data['RF_CUMM'] += probs
    training_data['RF_CUMM'] /= iterations
    print(probs.mean())
    # this helps
    training_data.loc[training_data['TARGET_EVENT'] == 'E', 'NN'] = 0
    training_data.loc[training_data['TARGET_EVENT'] == 'E', 'RF_CUMM'] = 0
    # 0.01836438758875496


    print(f"Logreg : { log_loss(training_data['TARGET_EVENT_BINARY'], training_data['LOGISTIC_REG'])}")
    # print(f"XGB+ : {log_loss(training_data['TARGET_EVENT_BINARY'], training_data['ENSEMBLE'])}")
    # print(f"RF+ : {log_loss(training_data['TARGET_EVENT_BINARY'], training_data['ENSEMBLE_NN'])}")
    # print(f"XGB : {log_loss(training_data['TARGET_EVENT_BINARY'], training_data['XGB'])}")
    print(f"RF : {log_loss(training_data['TARGET_EVENT_BINARY'], training_data['NN'])}")

    # print(f"LGBM : {log_loss(training_data['TARGET_EVENT_BINARY'], training_data['LGBM'])}")
    values = [31, 69, 0]
    values = values/np.sum(values)
    print(values)
    custom = training_data['LOGISTIC_REG_CUMM']*values[0] + training_data['RF_CUMM']*values[1]+training_data['XGB']*values[2]

    print(f"CUSTOM+ : {log_loss(training_data['TARGET_EVENT_BINARY'], custom)}")

    training_data['CUSTOM'] = custom
    _ = create_submission_file(training_data, 'CUSTOM', data_submission_example, filename='./predictions/nn-xgb-log-custom-unscaled.csv')
    new_proba = scale_yearly_proba(training_data, 'CUSTOM', targets =[ 0.0049, 0.0101, 0.0185], logging=True)
    training_data['CUSTOM'] = new_proba

    submission = create_submission_file(training_data, 'CUSTOM', data_submission_example, filename=f'./predictions/{params["RUN_NAME"]}.csv')