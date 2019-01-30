
Rossmann Cluster Project (Ref: Rossmann Kaggle Competition)

extract_csv_files.py (InputFile)
  Extract train, store, states of stores from csv files
  and then makes train_data.pickle, store_data.pickle
prepare_feature.py (SupervisedFeature)
  With train_data.pickle and store_data.pickle as an input, makes les.pickle and
  feature_train_data.pickle
models.py (Model_Util, NN_with_EntityEmbedding, NN_with_EntityEmbedding_Loading)
  train_test_model.py (TRAIN_DRIVER)
inference.py (Inference)
cluster_feature.py (ClusterFeature): 
  As an input, a supervised form of feature is given and
  then generate store_index and sales of month 1 to 12
cluster_models.py (ClusterModels):
  With a cluster feature pickle file given, make KMean and GMM
  cluster models and then show performance of both.
  Using TSNE algorithm, they are visualized  
