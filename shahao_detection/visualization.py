import matplotlib.patches as patches
import matplotlib.pyplot as plt

def visualization(img,box,label):
    plt.imshow(img)
    ax = plt.gca()
    # 默认框的颜色是黑色，第一个参数是左上角的点坐标
    # 第二个参数是宽，第三个参数是长
    ax.add_patch(plt.Rectangle((100, 200), 200, 100, color="blue", fill=False, linewidth=1))

    # 第三个参数是标签的内容
    # bbox里面facecolor是标签的颜色，alpha是标签的透明度
    ax.text(100, 200, "label 0.9", bbox={'facecolor':'blue', 'alpha':0.5})
    plt.savefig("./a.jpg")
    # plt.show()