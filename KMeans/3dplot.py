# 3D plot code starts
from mpl_toolkits import mplot3d
finalDf['color'] = ''
for i in range(0, len(finalDf)):
    finalDf["color"][i] = colors[int(finalDf['CLASS'][i])]
threedee = plt.figure().gca(projection='3d')
threedee.scatter(finalDf.index, finalDf['principal component 1'], finalDf['principal component 2']
                     , c = finalDf['color'])
threedee.set_xlabel('index')
threedee.set_ylabel('component1')
threedee.set_zlabel('component2')
plt.show()
# 3D plot code ends
