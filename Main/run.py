import os
import sys
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录添加到sys.path中
sys.path.append(project_root)
import config
import model_train
from model_test import Test as modelTest
import datetime
def _runTrain():
    print('\nStarting the training process......\n')

    train_instance = model_train.Train()
    print('Environments built successfully.\n')
    print('Size of train dataset:', train_instance.train_dataset_size)
    print('\n code size:', train_instance.origin_code_vocab_size)
    print('\n nl size:', train_instance.origin_nl_vocab_size)
    print('\n nl size:', train_instance.origin_pdg_vocab_size)


    print('\nStart training......\n')
    train_instance.run_train()
    print('\nTraining is done.')


def _runTest(checkpoint_name, save_path):
    print('\nInitializing the test environments......')

    test_instance = modelTest()
    test_instance.run_test_tmp(modelOrcheckName = checkpoint_name, test_save_path=save_path)
    # print('Environments built successfully.\n')
    # print('Size of test dataset:', test_instance.dataset_size)
    #
    # print('\nStart Testing......')
    # testSavePath=config.test_save_googleBleu4
    # test_instance.test_iter(testSavePath)
    # print('Testing is done.')

if __name__ == '__main__':
    _runTrain()

    print("=================================== test in model_google_best_bleu4 ============================================")
    print("The current time:{}".format(datetime.datetime.now()))
    checkpoint_name = "checkpoints_max_google_bleu4.pth"
    _runTest(checkpoint_name,config.test_save_googleBleu4)
