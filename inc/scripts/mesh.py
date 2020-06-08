# -*- coding: utf-8 -*-
import numpy as np


def build_line_2(r_x, R, Theta, graph):
    # Функция добавления учета границы области в граф.
    #
    # Аргументы:
    #   r_x -- радиус границы области
    #     R -- радиусы концентрических окружностей в пространстве (r, z)
    # Theta -- углы лучей, исходящих из начала координат

    '''
    x^2 + y^2 = r^2
    y = sqrt(r^2 - x^2)
    '''
    R_circ_x_idx = np.min((R >= r_x).nonzero())

    # радиусы окружностей, которые пересекаются с границей
    R_circ_x = R[R_circ_x_idx:]

    '''
    y = tan theta * x

    r <= R -> r^2 <= R^2

    r^2 = x^2 + y^2
    '''
    z = np.tan(Theta) * r_x
    R__2 = r_x**2 + z**2 
    Theta_line_x_idx, = (R__2 <= R[-1]**2).nonzero()

    # углы лучей, которые пересекаются с границей
    Theta_line_x = Theta[Theta_line_x_idx]

    '''
    cos(theta) = r / R
    R = r / cos(theta)
    '''
    R_ = R[R_circ_x_idx-1:]

    # радиусы точек пересечений границы с лучами
    R_line_x = r_x / np.cos(Theta_line_x)

    # радиусы ближайших к границе точек сетки
    R_circ_x_idx_near_idx = \
        np.argmin(np.abs(np.repeat([R_], len(R_line_x), axis=0)
                         - R_line_x.reshape(len(R_line_x), 1)), axis=1)

    # индекс ближайших к границе узлов
    verts_near_idx = (R_circ_x_idx_near_idx + R_circ_x_idx-1) \
                   + len(R) * Theta_line_x_idx

    verts = graph['vertices']
    r, z = verts[:, 0], verts[:, 1]

    # поместить ближашие к границе узлы на границу
    r[verts_near_idx] = r_x


    R_circ1_x_idx = np.delete(np.arange(R_circ_x_idx-1, len(R)),
                              np.append(R_circ_x_idx_near_idx, 0))
    R_circ1_x = R[R_circ1_x_idx]

    # пересечения границы с верхними частями окружностей
    z_circ1_x_positive = np.sqrt(R_circ1_x**2 - r_x**2)
    
    # пересечения границы с верхними частями окружностей (возвращает view)
    z_circ1_x_negative = np.flip(-z_circ1_x_positive, axis=0)
    
    # пересечения границы с окружностями
    z_circ1_x = np.concatenate([z_circ1_x_negative, z_circ1_x_positive])

    r_x_ = np.full(z_circ1_x.shape, r_x)
    verts_circ1_x = np.stack([r_x_, z_circ1_x], axis=1)

    verts_x = np.concatenate([verts, verts_circ1_x])
    verts_circ1_x_idx = np.arange(len(verts_circ1_x)) + len(verts)

    z_x = np.concatenate([z[verts_near_idx], z_circ1_x], axis=0)
    idx = np.concatenate([verts_near_idx, verts_circ1_x_idx], axis=0)
    dtype = [('z_x', float), ('idx', int)]

    z_x_idx = np.empty(len(z_x), dtype=dtype)
    z_x_idx['z_x'] = z_x
    z_x_idx['idx'] = idx

    verts_x_idx = np.sort(z_x_idx, order='z_x')['idx']

    segs_line_x = np.stack([verts_x_idx[:-1], verts_x_idx[1:]], axis=1)
    
    segs_x = np.concatenate([graph['segments'], segs_line_x], axis=0)


    graph_x = dict(vertices=verts_x, segments=segs_x)

    return graph_x


def calc_mesh(r, R, Theta):
    # Функция расчета сетки
    # с учетом границ областей в системе скважина-пласт.
    #
    # Аргументы:
    #     r -- радиусы границы области
    #     R -- радиусы концентрических окружностей в пространстве (r, z)
    # Theta -- углы лучей, исходящих из начала координат

    import triangle as tr
    from others import (cartprod, cylinder_verts,
                        tri2mesh, concat_graphs)


    # точка в начале координат

    verts0 = [[0, 0]]
    graph0 = dict(vertices=verts0)


    # граф с точками, регулярно (структуировано) расположенными в пространстве

    RT_verts1 = cartprod(R, Theta, order='F')  # декартово произведение
    verts1 = cylinder_verts(RT_verts1)  # перевод из полярных координат в декартовые
    graph1 = dict(vertices=verts1, segments=np.array([]).reshape(0, 2))


    if r is None or len(r) == 0:
        # в случае если не заданы границы областей
        graph1 = dict(vertices=np.array([]).reshape(0, 2),
                      segments=np.array([]).reshape(0, 2))
    else:
        # добавление учета границ областей в графе
        for r_ in r:
            graph1 = build_line_2(r_, R, Theta, graph1)

    # объединение графов
    graph = concat_graphs(graph0, graph1)

    # добавление в граф границы выпуклой оболочки множества точек триангуляции,
    # служит границей расчетной сетки
    hull_segments = tr.convex_hull(graph['vertices'])
    graph['segments'] = np.append(graph['segments'], hull_segments, axis=0)

    # триангуляция Делоне по заданному графу
    tri = tr.triangulate(graph, 'p')

    # перевод расчетной сетки в читаемый FEniCS формат
    mesh = tri2mesh(tri)

    return graph, tri, mesh


def main():
    from others import crange


    # r_s  -- радиус скважины
    # r_pz -- радиус промытой зоны
    # r_zp -- радиус зоны проникновения

    r_s, r_pz, r_zp = 1, 2.2, 5.8


    #        n_R_2 -- кол-во радиусов концентрических окружностей
    #                 в единицу логарифма длины в пространстве (r / 2, z / 2)
    #      n_Theta -- кол-во углов лучей, исходящих из начала координат
    #                 в правом полупространстве (r / 2, z / 2)
    # log_R_2_min,
    # log_R_2_max  -- логарифмы значений границ радиуса
    #                 для концентрических окружностей

    n_R_2, n_Theta = 2*10, 2*12
    log_R_2_min, log_R_2_max = -1.5, 4


    R_2 = 10 ** crange(log_R_2_min, log_R_2_max, 1/n_R_2)
    R = 2 * R_2

    Theta = np.linspace(-np.pi/2, np.pi/2, n_Theta)

    # построение расчетной сетки
    graph, tri, mesh = calc_mesh([r_s, r_pz, r_zp], R, Theta)

    # вывод кол-ва узлов расчетной сетки
    print(f'{mesh.num_vertices():d} узлов\n')


    # вывод графиков графа и триангуляции

    import matplotlib.pyplot as plt
    from others import mpl_init, plot_mesh
    mpl_init()


    plt.figure()
    plt.subplot(
        aspect='equal',
        adjustable='box',
        xlabel=r'$r / R_\text{с}$',
        ylabel=r'$z / R_\text{с}$'
    )

    plt.grid()
    plot_mesh(graph, tri)


    plt.figure()
    plt.subplot(
        aspect='equal',
        adjustable='datalim',
        xlabel=r'$r / R_\text{с}$',
        ylabel=r'$z / R_\text{с}$',
        xlim=(0, 7),
        ylim=(-1, 1),
    )

    plt.grid()
    plot_mesh(graph, tri)


    plt.show()


if __name__ == "__main__":
    main()