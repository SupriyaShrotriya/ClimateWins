# Objective:
ClimateWins is a European nonprofit organization, that is interested in using machine learning to help predict the consequences of climate change around Europe and, potentially, the world.

# Key Questions Include
1.How is machine learning used? Is it applicable to weather data?
2.Can machine learning reliably predict whether the weather will be pleasant based on historical weather data?

# Hypotheses:
1.Weather prediction accuracy may change depending on the location and climate. 
2.Machine learning can help identify signs of climate change and its potential effects.
3.Supervised learning can be used to predict if a day will be pleasant or not based on historical weather data.

# Data 
Data was collected between 1800s to 2022 by ‘European Climate Assessment and Data Set Project’ found at https://www.ecad.eu/
Data consisted of temperature, wind speed, snow, and global radiation from 18 different weather stations.

Machine Learning Algorithms:
- Gradient Descent Optimization
- K-Nearest Neighbor Algorithm (KNN)
- Artifical Neural Network (ANN)
- Decision Tree

# Tools

For this project, the following python libraries were used:
- pandas, numpy, matplotlib, matplotlib.pyplot, os, operator
- sklearn: .preprocessing, .metrics, .neural_netwrok, MLPCLassifier, .model_selection, train_test_split, .ensemble, tree import plot_tree, .model_selection
- RandomForestClassifier, GridSearchCV, argmax, metrics
- multilabel_confusion_matrix, accuracy_score, ConfusionMatrixDisplay, StandardScaler
- tensorflow, keras (keras.models, keras.layers with LSTM), Sequential, Conv1D, Conv2D, Dense, Dropout, BatchNormalization, Flatten, MaxPooling1D
- scipy: .cluster.hierarchy import dendrogram, linkage, fcluster
- bayes_opt, BayesianOptimization, LeakyReLU, BatchNormalization

