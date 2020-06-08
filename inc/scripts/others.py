import numpy as np
import matplotlib.pyplot as plt


def crange(a, b, d, **kw):
    n = int((b - a) / d)
    arr = np.empty(n + 1, **kw)
    arr[:n] = np.arange(a, b, d, **kw)
    arr[n] = b
    
    return arr


def cartprod(*arrays, **kw_reshape):
    # декартово произведение

    N = len(arrays)
    
    cart_product = np.transpose(np.meshgrid(*arrays, indexing='ij'),
                                np.roll(np.arange(N + 1), -1)).reshape(-1, N, **kw_reshape)

    return cart_product


def tri2mesh(tri):
    from fenics import Mesh, MeshEditor

    mesh = Mesh()
    editor = MeshEditor()

    c_str = 'triangle'
    tdim = gdim = 2
    editor.open(mesh, c_str, tdim, gdim)

    editor.init_vertices(len(tri['vertices']))
    editor.init_cells(len(tri['triangles']))

    for i, vert in enumerate(tri['vertices']):
        editor.add_vertex(i, np.array(vert))

    for i, cell in enumerate(tri['triangles']):
        editor.add_cell(i, np.array(cell))

    editor.close()

    return mesh


def cylinder_verts(rt_verts):
    # перевод узлов rt_verts из полярных координат в декартовые

    xy_verts = np.empty_like(rt_verts)

    r, theta = rt_verts[:,0], rt_verts[:,1]

    x, y = r * np.cos(theta), r * np.sin(theta)

    # исправить cos(pi/2) и cos(-pi/2), чтобы выровнить границу расчетной сетки
    x[theta == np.pi / 2] = 0
    x[theta == - np.pi / 2] = 0
    
    xy_verts[:,0], xy_verts[:,1] = x, y
    
    return xy_verts


def concat_graphs(*graphs):
    # объединение графов в graphs

    if isinstance(graphs[0], (list, tuple)):
        graphs = graphs[0]

    res_graph = {}

    if np.any(['vertices' in graph for graph in graphs]):
        res_graph['vertices'] = np.array([]).reshape(0,2)

    if np.any(['segments' in graph for graph in graphs]):
        res_graph['segments'] = np.array([]).reshape(0,2)

    for graph in graphs:
        if 'segments' in graph:
            offset = len(res_graph['vertices'])
            segs = np.asarray(graph['segments']) + offset
            res_graph['segments'] = np.concatenate([res_graph['segments'], segs])

        if 'vertices' in graph:
            res_graph['vertices'] = np.concatenate([res_graph['vertices'], graph['vertices']])

    return res_graph


def mpl_init():
    from matplotlib import rc
    rc('text', usetex=True)

    plt.rcParams.update({'font.size': 14})

    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage[english,russian]{babel}',
        r'\usepackage{amsmath}',
        r'\usepackage{siunitx}'
    ]

    plt.rcParams["text.latex.preview"] = True


def plot_mesh(graph, tri, ax=None):
    # функция вывода графиков графа и триангуляции

    if ax == None:
        ax = plt.gca()

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    verts = np.asarray(graph['vertices'])
    idx_segs = np.asarray(graph['segments'], dtype=np.int)

    # plot segments
    segs = np.array([]).reshape(0, 2, 2)
    for idx_segm in idx_segs:
        segs = np.append(segs, [verts[idx_segm]], axis=0)

    from matplotlib import collections as mc
    lc = mc.LineCollection(segs, colors='r', linewidths=4, zorder=2)#, linewidths=2)
    ax.add_collection(lc)

    # plot vertices
    ax.plot(verts[:, 0], verts[:, 1], '.', c=prop_cycle[0], zorder=3, rasterized=False)

    # plot mesh
    verts = np.asarray(tri['vertices'])
    triangles = tri['triangles']

    ax.triplot(verts[:, 0], verts[:, 1], triangles,
                c=prop_cycle[0], zorder=2, rasterized=False)