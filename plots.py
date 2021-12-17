
import matplotlib.pyplot as plt



def produce_plot(name, results, gamma, accuracy ):
    loadPath = 'model/' + name + '.pkl'
    savePath = './plots/' + name + '.png'
    # results = pickle.load(open('model/Gamma0.2-Batch128-SGD(0.001,0.99).pkl', 'rb'))
    plt.plot(results['loss'])
    plt.plot(results['clustering_loss'])
    plt.plot(results['decoder_out_loss'])
    plt.legend(['Total Loss', 'Clustering Loss (KLD)', 'AE Loss (MSE)'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Gamma = ' + str(gamma) +
              ', Acc = ' + str(accuracy))
    plt.savefig(savePath)
    plt.clf()

    savePath = './plots/' + name + 'Accuracy' + '.png'
    plt.plot(results['clustering_accuracy'])
    plt.legend(['Accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Ratio')
    plt.title('Gamma = ' + str(gamma))
    plt.savefig(savePath)
    plt.clf()
