




import pandas as pd
import numpy as np

# boston.csv
gdrive_data_url = "https://drive.google.com/file/d/1cnIxAAqXnG9vHO59EuRBbnit_3rM_UOT/view?usp=share_link"
file_id = gdrive_data_url.split('/')[-2]
data_file='https://drive.google.com/uc?export=download&id=' + file_id
df = pd.read_csv(data_file)
df.drop("Unnamed: 0", axis=1, inplace=True)
df.columns = df.columns.str.upper()
df.rename(columns={'BLACK':'B', 'MEDV':'Price'}, inplace=True)

X = df.drop(columns=['Price'])
y = df[['Price']]
y['Price'] = y['Price'].apply(lambda x: 1 if x > 30 else 0)

y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape



from pycaret.classification import ClassificationExperiment

train_data = pd.concat([X_train, y_train], axis=1)

exp = ClassificationExperiment(
    target="Price",
    fold = 5,
    session_id=13,
    normalize=True,
    log_experiment = True,
    verbose = True
).fit(train_data)

pd.DataFrame(list(exp.get_params().items()), columns=['Parameter', 'Value'])

pd.DataFrame(list(exp._fit_state.items()), columns=["Description", "Value"])

"""# 3. Compare Baseline"""

exp.models()

result = exp.compare_models(errors="raise")

leaderboard = result.leaderboard
model = result.models
best_model = result.best

leaderboard

model

best_model

tuned_result = exp.tune_model(best_model, n_iter=50, optimize="f1")

tuned_result



rf = exp.create_model("rf")

"""# 5. Tune Hyperparameters"""

tuned_rf = exp.tune_model(rf.pipeline, n_iter=50, optimize="f1").pipeline

rf

tuned_rf



en_rf = exp.create_model('rf')

bagged_rf = exp.ensemble_model(en_rf.pipeline, n_estimators=50).pipeline

bagged_rf

boosted_rf = exp.ensemble_model(en_rf.pipeline, method="Boosting").pipeline

boosted_rf



from pycaret.plots.classification import class_distribution, confusion_matrix, roc_curve
from pycaret.plots.feature import permutation_importance
from pycaret.plots.regression import prediction_error

confusion_matrix(rf.pipeline, X_test, y_test).show()

confusion_matrix(tuned_rf, X_test, y_test).show()

roc_curve(rf.pipeline, X_test, y_test).show()

prediction_error(rf.pipeline, exp.X_test, exp.y_test).show()

class_distribution(exp.y_test).show()

"""# 8. Interpret Model

## Via `Pycaret`
"""

from pycaret.plots.feature import permutation_importance, shap_summary, \
shap_beeswarm, partial_dependence, ice_curve

permutation_importance(rf.pipeline, exp.X_train, exp.y_train).show()

fig_beeswarm = shap_beeswarm(rf.pipeline, exp.X_test)
fig_beeswarm.show()

fig_summary = shap_summary(rf.pipeline, exp.X_test)
fig_summary.show()

fig_pdp = partial_dependence(rf.pipeline, exp.X_test, feature="RM")
fig_pdp.show()

# Two-feature interaction Partial Dependence
fig_pdp_2d = partial_dependence(rf.pipeline, exp.X_test, feature=("RM", "LSTAT"))
fig_pdp_2d.show()

fig_ice = ice_curve(rf.pipeline, exp.X_test, feature="RM")
fig_ice.show()



import shap

estimator = rf.pipeline.steps[-1][1]
preprocessor = rf.pipeline[:-1] if len(rf.pipeline.steps) > 1 else None

X_test_transformed = preprocessor.transform(X_test) if preprocessor else X_test

# Reattach feature names
if preprocessor:
    feature_names = preprocessor.get_feature_names_out()
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
else:
    feature_names = X_test.columns

explainer = shap.Explainer(estimator, X_test_transformed)
shap_values = explainer(X_test_transformed)

# Extract positive class for binary classification models
if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

shap.plots.bar(shap_values)

rm_idx = list(X_test_transformed.columns).index("RM")
shap.plots.scatter(shap_values[:, rm_idx], color=shap_values)

# Extract expected value for positive class
if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_val = explainer.expected_value[1]
else:
    base_val = explainer.expected_value

shap.initjs()

shap.force_plot(
    base_val,
    shap_values.values,
    X_test_transformed
)

shap.plots.beeswarm(shap_values)

shap.plots.waterfall(shap_values[11])



from sklearn.metrics import classification_report, confusion_matrix

best_pred_res = exp.predict_model(best_model, data=X_test)
best_pred_res.predictions.head()

best_labels = best_pred_res.predictions["prediction_label"]
print("--- Best Model Metrics ---")
print(confusion_matrix(y_test, best_labels))
print(classification_report(y_test, best_labels))

rf_pred_res = exp.predict_model(rf.pipeline, data=X_test)
rf_labels = rf_pred_res.predictions["prediction_label"]

print("--- Random Forest Metrics ---")
print(confusion_matrix(y_test, rf_labels))
print(classification_report(y_test, rf_labels))

"""# 10. Save / Load Model"""

exp.save_model(rf, 'rf.model')

loaded_bestmodel = exp.load_model("rf.model")
print(loaded_bestmodel)

