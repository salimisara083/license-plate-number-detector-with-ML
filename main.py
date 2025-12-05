from dataset import *
from model import Model
from pelak import Pelak
PATH = r"C:\Users\lenovo\10mlprojects\pelak\dataset\pelak.png"

def main():
    x_train_scaled, x_test_scaled, y_train, y_test = clean_scale_dataset(create_dataset()[0], create_dataset()[1])
    model = Model(x_train_scaled, y_train)
    pelak = Pelak(PATH)
    model.pred(pelak.segments())
    model.model_accuracy(y_test, x_test_scaled)
    pelak.write(model.acc, model.predictions)
    
if __name__ == "__main__" :
    main()


