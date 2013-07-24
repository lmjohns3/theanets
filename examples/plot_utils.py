import matplotlib.pyplot as plt
import numpy as np

def make_example_plot(data, use_axis, reshape_size=None, plotdim=9, randomize=True, title=None, savename=None, display_now=True):
    #Default is a 9x9 plot showing the learned basis functions as 28x28 images.
    if reshape_size == None:
        other_dim = 0 if use_axis == 1 else 1
        r = int(np.sqrt(data.shape[other_dim]))
        reshape_size = (r,r)
        print "Reshape size is",reshape_size
    if randomize==True:
        ind = np.random.randint(data.shape[use_axis],size=(plotdim,plotdim))
    else:
        ind = np.linspace(0,data.shape[use_axis],endpoint=False,num=plotdim*plotdim).reshape(plotdim,plotdim)
        ind.round()

    _, axarr = plt.subplots(plotdim,plotdim)
    for i in range(plotdim):
        for j in range(plotdim):
            axarr[i,j].get_xaxis().set_visible(False)
            axarr[i,j].get_yaxis().set_visible(False)
            axarr[i,j].set_frame_on(False)
            #This may be overkill, since most data will be 2D
            #Fancy indexing to slice across one dimension
            slicer = [slice(None)]*data.ndim
            slicer[use_axis] = ind[i,j]
            axarr[i,j].imshow(data[tuple(slicer)].reshape(*reshape_size), cmap=plt.cm.gray)
    if title != None:
        plt.suptitle(title)
    plt.tight_layout()
    if savename != None:
        plt.savefig(savename, bbox_inches=0)
    if display_now == True:
        plt.show()

def plot_autoencoder_experiment(e, testing_data):
    weights = w = []
    min_size = 1e100
    for i in range(len(e.network.weights)):
        w.append(e.network.weights[i].get_value())
        #w.append(.5*wi)

    min_size = min([x.shape[1] for x in w])
    code_layer_index = max([i for i,x in enumerate(w) if x.shape[1] == min_size])
    print "Code layer index is",code_layer_index

    dim = 4
    display_flag = False
    make_example_plot(testing_data, 0, plotdim=dim, title="Sample data", savename="input_data.png", display_now=display_flag, randomize=False)

    for i in w:
        print "Layer shape",i.shape

    processed = testing_data
    for i in w:
        processed = np.dot(processed,i)

    if code_layer_index == len(w)-1:
        print "Code layer is the last layer in w, adding reconstruction. This probably means that tied_weights=True"
        for i in w[::-1]:
            processed = np.dot(processed,i.T)

    print "Learned filters",w[code_layer_index].shape
    make_example_plot(w[0], 1, plotdim=dim, title="First layer learned filters", savename="f_layer_filters.png", display_now=display_flag, randomize=False)

    print "Feedforward output shape", processed.shape
    make_example_plot(processed, 0, plotdim=dim, title="Coded and reconstructed output", savename="reconstructed_data.png", display_now=display_flag, randomize=False)

    plt.show()


def plot_classifier_experiment(e, testing_data):
    weights = w = []
    min_size = 1e100
    for i in range(len(e.network.weights)):
        w.append(e.network.weights[i].get_value())
        #w.append(.5*wi)

    min_size = min([x.shape[1] for x in w])
    code_layer_index = max([i for i,x in enumerate(w) if x.shape[1] == min_size])
    print "Code layer index is",code_layer_index

    dim = 4
    display_flag = False
    raw_data = testing_data[0]
    make_example_plot(raw_data, 0, plotdim=dim, title="Sample data", savename="input_data.png", display_now=display_flag, randomize=False)

    for i in w:
        print "Layer shape",i.shape

    processed = raw_data
    for i in w:
        processed = np.dot(processed,i)

    print "Learned filters",w[code_layer_index].shape
    make_example_plot(w[0], 1, plotdim=dim, title="First layer learned filters", savename="f_layer_filters.png", display_now=display_flag, randomize=False)

    plt.show()
