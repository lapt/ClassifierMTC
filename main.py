from classifier_evaluation.evaluation_clf import max_classification_random
from classifier_evaluation.TestHelper import evaluation
import time


start_time = time.time()


def create_cls_priority():
    # Ranking de los algoritmos con diferentes parametros  --> DATA ENTRENAMIENTO
    path_data = "training_data/TRAIN_INTER.tsv"  # Clean data
    path_result = "result/result_final.csv"  # Result final of every classifier
    path_result_cls_transport = "results_models/cls_prioridad.joblib"
    max_classification_random(path_data, path_result, path_result_cls_transport, 1000)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Finish")


def test_cls_priority():
    # Ranking de los modelos con parametros obtimos --> DATA PRUEBA
    path_corpus_train = 'training_data/TRAIN_INTER.tsv'
    path_corpus_test = 'training_data/TEST_INTER.tsv'
    evaluation(path_corpus_train, path_corpus_test)


if __name__ == '__main__':  # Only get better parameters
    test_cls_priority()
