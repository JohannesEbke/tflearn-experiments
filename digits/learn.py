from models import AutoEncoder

def learn():
    import tflearn.datasets.mnist as mnist
    X, Y, testX, testY = mnist.load_data(one_hot=True)

    d = AutoEncoder(784, [784*2], 64)
    d.learn(X, testX)
    d.save()

if __name__=="__main__":
    learn()
