import matplotlib.pyplot as plt


def plot_two_dimensions(title, data_array, legend_array, save_flag=False):
    if len(data_array) != 2 or 2 != len(legend_array):
        print("数据与标题个数不等于2")
        return

    # plot
    for data in data_array:
        plt.plot(data)
    plt.title(title)
    plt.legend(legend_array)

    # save
    if save_flag:
        plt.savefig("output\\" + title + ".png")
    plt.show()
