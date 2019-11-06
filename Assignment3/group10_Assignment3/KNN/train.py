#here importin the required libraries
import numpy as np
import glob
import classify

#*******************************************************************************************************************************************

source_path1="Train/kA"
source_path2="Train/kha"
source_path3="Train/khA"

source_path1t="Test/kA"
source_path2t="Test/kha"
source_path3t="Test/khA"


confusion_matrix=np.zeros((3,3));
d=39
k=32
classify.knn_classify(source_path1t,source_path1,source_path2,source_path3,confusion_matrix,k,d,0);

print(confusion_matrix);

classify.knn_classify(source_path2t,source_path1,source_path2,source_path3,confusion_matrix,k,d,1);

print(confusion_matrix);

classify.knn_classify(source_path3t,source_path1,source_path2,source_path3,confusion_matrix,k,d,2);

print(confusion_matrix);



