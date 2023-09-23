# Predicting-Credit-Risk

In this project I trained two models to predict whether a loan would be high-risk or low-risk and compared the differences. After preprocessing the data, I used a logistic regression model and a random forest classifier model, before and after scaling.

### My prediction before training:
I think the random forests classifier will be more accurate. The ensemble approach will be more robust and better suited to handle complex datasets.

**Logistic Regression:**
`classifier = LogisticRegression(max_iter=1000)`  
`classifier.fit (X_train_coded, y_train_coded)`  
`print(f"Training Data Score: {classifier.score(X_train_coded, y_train_coded)}")`  
`print(f"Testing Data Score: {classifier.score(X_test_coded, y_test_coded)}")`  
Training Data Score: 0.7044334975369458
Testing Data Score: 0.5686941726924712

**Random Forest Classifier:**
`rf_model = RandomForestClassifier(n_estimators=500, random_state=78)`  
`rf_model = rf_model.fit(X_train_coded, y_train_coded)`  
`print(f"Training Data Score: {rf_model.score(X_train_coded, y_train_coded)}")`  
`print(f"Testing Data Score: {rf_model.score(X_test_coded, y_test_coded)}")`  
Training Data Score: 1.0
Testing Data Score: 0.6427052318162484

### I then made a new prediction before scaling the data:
I think the random forests classifier will be even more accurate with scaled data for the same reasons; with better data the ensemble approach should work even better.

**Scaling Data:**
`scaler = StandardScaler().fit(X_train_coded)`  
`X_train_scaled = scaler.transform(X_train_coded)`  
`X_test_scaled = scaler.transform(X_test_coded)`  

**Scaled Logistic Regression:**
Training Data Score: 0.710919540229885
Testing Data Score: 0.7598894087622289

**Scaled Random Forest Classifier:**
Training Data Score: 1.0
Testing Data Score: 0.6439812845597618

### Results:
I was quite wrong, the logistic regression performed better. I should have considered its perfect training score which indicates overfitting. Other possible reasons are that the model's simplicity was an advantage for this dataset. It seems scaling really benefits logistic regressions while barely affecting random forest models in this case.