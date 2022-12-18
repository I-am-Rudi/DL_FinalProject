import torch
import matplotlib.pyplot as plt

def plot_data(data):
    train_data, train_target, test_data, test_target = data
    plt.rcParams['font.size'] = '16'
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_aspect( 1 )
    ax.add_artist( plt.Circle( (.5, .5 ), 1/(2*torch.pi)**(1/2), fill = False ) )
    plt.title( 'Sanity check for data_set' )
    for i in range(train_target.shape[0]):
        if train_target[i]:
            train1 = ax.scatter(train_data[i,0], train_data[i,1], color = "darkblue")
        else:
            train0 = ax.scatter(train_data[i,0], train_data[i,1], color = "darkblue", facecolors='none')
    for i in range(test_target.shape[0]):
        if test_target[i]:
            test1 = ax.scatter(test_data[i,0], test_data[i,1], color = "darkorange")
        else:    
            test0 = ax.scatter(test_data[i,0], test_data[i,1], color = "darkorange", facecolors='none')
    empty = ax.scatter(1, 1, color = "none")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #ax.grid()
    #ax.grid(which="minor", linestyle=':', lw=.5)
    ax.tick_params(axis="both", direction="in", top = True, right=True, which="both")
    ax.legend([empty, train1, train0, empty, test1, test0], ["Training Data (size: " + str(train_data.shape[0]) + "):", "Target: 1",  "Target: 0" , "Test Data (size: " + str(test_data.shape[0]) + "):",  "Target: 1",  "Target: 0" ], prop={"size":14}, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    fig.tight_layout()
    plt.show()

def plot_analysis(mean, std, epochs, nb_trials, name):

    fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi = 80)
    plt.rcParams['font.size'] = '16'

    x = range(1, epochs+1)

    mean0, = ax.plot(x, mean, color="darkblue")
    std1, = ax.plot(x, mean+std, color="blue", alpha=0.97)
    std2, = ax.plot(x, mean-std, color="blue", alpha=0.97)
    fill = ax.fill_between(x, mean+std, mean-std, color="blue", alpha=0.5)

    ax.set_title("Mean and standard deviation of test_accuracy after" + '\n' + "{} runs with {} epochs".format(nb_trials, epochs))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.grid()
    ax.grid(which="minor", linestyle=':', lw=.5)
    ax.tick_params(axis="both", direction="in", top = True, right=True, which="both")
    ax.legend([mean0, (std1, fill ,std2)], ["mean", "standard deviation"], prop={"size":14})
    fig.tight_layout()
    fig.savefig("./Plots/" + "Analysis_" + name.replace(' ', '_') + ".png")