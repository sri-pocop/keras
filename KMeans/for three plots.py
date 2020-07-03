from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
s_s = StandardScaler()
for_3d_plot = pd.DataFrame(data = (x), columns = ['p1', 'p2','p3'])
for_3d_plot = pd.concat([for_3d_plot, dataset[['CLASS']]], axis = 1)
for_3d_plot['color'] = ''
for i in range(0, len(for_3d_plot)):
    for_3d_plot["color"][i] = colors[int(for_3d_plot['CLASS'][i])]
threedee = plt.figure().gca(projection='3d')
threedee.scatter(for_3d_plot.index, for_3d_plot['p1'], for_3d_plot['p2']
                     , c = for_3d_plot['color'])
threedee.set_xlabel('index')
threedee.set_ylabel('p1')
threedee.set_zlabel('p2')

threedee = plt.figure().gca(projection='3d')
threedee.scatter(for_3d_plot.index, for_3d_plot['p2'], for_3d_plot['p3']
                     , c = for_3d_plot['color'])
threedee.set_xlabel('index')
threedee.set_ylabel('p2')
threedee.set_zlabel('p3')

threedee = plt.figure().gca(projection='3d')
threedee.scatter(for_3d_plot.index, for_3d_plot['p1'], for_3d_plot['p3']
                     , c = for_3d_plot['color'])
threedee.set_xlabel('index')
threedee.set_ylabel('p1')
threedee.set_zlabel('p3')
