from models import AutoEncoder

def learn():
    import tflearn.datasets.mnist as mnist
    X, Y, testX, testY = mnist.load_data(one_hot=True)

    for i in range(1, 65):
        d = AutoEncoder(784, [256], i)
        d.learn(X, testX)
        d.save()

if __name__=="__main__":
    learn()
