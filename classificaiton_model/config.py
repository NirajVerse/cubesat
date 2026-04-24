
DATASET_PATH = '/work/users/ng739/cubesat/model/cubesat/data/ohid_ff'
IMAGE_SIZE = 512  
RESIZE_TO = 224   
TRAIN_SIZE = 597  
TEST_SIZE = 600   


BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_RUNS = 3


MODELS = [
    'resnet18',
    'resnet50',
    'vgg16',
    'logistic',
    'mobilenetv2',
    'densenet121',
    'shufflenetv2'
    #'inceptionv3'
]


AUGMENTATION = True


CHECKPOINT_DIR = './checkpoints/'
RESULTS_DIR = './results/'
