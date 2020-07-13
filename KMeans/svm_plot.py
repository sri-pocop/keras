h = .02  # step size in the mesh

C = 1.0  # SVM regularization parameter

# create a mesh to plot in
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
z_min, z_max = x[:, 2].min() - 1, x[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
xx, zz = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(z_min, z_max, h))
clf = svc
for i, clf in enumerate(([0,1],[0,2],[1,2])):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2,3, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = scv.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=.8)

    # Plot also the training points
    plt.scatter(x[:, clf[0]], x[:, clf[1]], c=y)#, cmap=plt.cm.coolwarm)
    plt.xlabel('feature ' + str(clf[0]))
    plt.ylabel('feature ' + str(clf[1]))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('feature' + str(clf[0]) + '&' + str(clf[1]) )
    print(i,clf)
    plt.show()
plt.show()
