import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import igraph as ig
import dgl
import sys
from matplotlib import pyplot as plt
import matplotlib
import torch
import os

sys.path.append("../utils/")

from constants import DatasetType, GraphVisualizationTool, network_repository_cora_url, cora_label_to_color_map
from utils import convert_adj_to_edge_index


def draw_picture(g,type_human,type_scene,i=0):
    # draw_picture(g)
    # hg = dgl.to_homogeneous(g)
    # link = hg.adj()
    # print(link)
    # aa = link._indices().numpy()
    # aa = aa.T
    # aa = aa.astype(np.int32)
    # np.savetxt('link.txt', aa, fmt='%d')
    # For scene_ID, 0: Lane, 1: Sidewalk, 2: Lawn, 3: Obstacle, 4: Door
    # color = [[88, 88, 88], [255, 127, 38], [14, 209, 69], [235, 28, 36], [0, 0, 254]]
    save_dir_base = './relation_visual'+ str(i)
    if not os.path.exists(save_dir_base):
        os.makedirs(save_dir_base)

    hg = g.to('cpu')
    color = {'lane': [88, 88, 88], 'sidewalk': [255, 127, 38], 'lawn':[14, 209, 69], 'Obstacle': [235, 28, 36], 'Door': [0, 0, 254]}
    bb = hg.nodes['lawn'].data['picture'][1]
    h,w = hg.nodes['lawn'].data['picture'][1].shape
    scene_picture = np.ones((h,w,3),np.uint8) * -1

    for ts in type_scene:
        for i in range(hg.nodes[ts].data['picture'].shape[0]):
            if(h == hg.nodes[ts].data['picture'][i].shape[0]):
                picture_data = hg.nodes[ts].data['picture'][i].cpu().numpy()
                mask = np.where(picture_data==1)
                scene_picture[mask[0],mask[1],:] = color[ts]
    # plt.imshow(scene_picture).astype(np.uint8)
    # scene_picture = scene_picture.transpose(1,0,2).astype(np.uint8)
    scene_picture = scene_picture.astype(np.uint8)
    # matplotlib.image.imsave('picture.png',scene_picture)
    #
    colors = ['b','g','r','c','m','y','k','w']
    for srctype, etype, dsttype in hg.canonical_etypes:
        if etype!='adj':
            continue
        # for i in range(0, g.num_edges((srctype,'adj',dsttype)), 10):
        agent = hg.find_edges(torch.arange(0,min(8,g.num_edges((srctype,'adj',dsttype)))),(srctype,'adj',dsttype))
        plt.imshow(scene_picture)
        for i in range(len(agent[0])):
            src_data = hg.nodes[srctype].data['x'][agent[0][i]].view(8,2)
            plt.scatter(src_data[:,0],src_data[:,1],c=colors[i],marker='o')
            plt.text(src_data[0,0],src_data[0,1],"s {} {}".format(srctype,i),fontsize=7,color='black')
            src_data = hg.nodes[srctype].data['y'][agent[0][i]]
            plt.scatter(src_data[:, 0], src_data[:, 1], c=colors[i], marker='x')

            dst_data = hg.nodes[dsttype].data['x'][agent[1][i]].view(8,2)
            plt.scatter(dst_data[:, 0], dst_data[:, 1], c=colors[i], marker='o')
            plt.text(dst_data[0, 0], dst_data[0, 1], "d {} {}".format(dsttype, i), fontsize=7, color='black')
            dst_data = hg.nodes[dsttype].data['y'][agent[1][i]]
            plt.scatter(dst_data[:, 0], dst_data[:, 1], c=colors[i], marker='x')

        plt.savefig(save_dir_base + '/' + srctype +'_'+ dsttype + '.png',bbox_inches='tight', dpi=900)
        plt.close()

        # agent_all = torch.agent[0]

def draw_predict(g, predicted_future, data_scale, run, type_scene, type_human,i, mean_test, pic_b, pic_dict_b):

    save_dir_base = './result_visual/' + str(run) + '/' + str(i)
    if not os.path.exists(save_dir_base):
        os.makedirs(save_dir_base)

    hg = g.to('cpu')
    for human in type_human:
        hg.nodes[human].data['x'] = hg.nodes[human].data['x'].view(-1,8,2)
        hg.nodes[human].data['x'] /= data_scale
        hg.nodes[human].data['x'] +=  mean_test[i][human]
        predicted_future[human] /= data_scale
        predicted_future[human] += mean_test[i][human]
        hg.nodes[human].data['y'] /= data_scale
        hg.nodes[human].data['y'] += mean_test[i][human]

    colors = ['b','g','r','c','m','y','k','w']

    for scene in pic_dict_b.keys():
        plt.imshow(pic_b[scene])
        scene_dict = pic_dict_b[scene]
        for human in scene_dict.keys():
            index_num = scene_dict[human]
            for index in index_num:
                traj_data = hg.nodes[human].data['x'][index]
                plt.scatter(traj_data[:, 0], traj_data[:, 1], c='b', marker='o')
                plt.text(traj_data[0, 0], traj_data[0, 1], "s {} {}".format(human, i), fontsize=7, color='black')
                traj_data = hg.nodes[human].data['y'][index]
                plt.scatter(traj_data[:, 0], traj_data[:, 1], c='r', marker='x')
                traj_data = predicted_future[human][index, :, :]
                plt.scatter(traj_data[:, 0], traj_data[:, 1], c='g', marker='v')


    # for srctype, etype, dsttype in hg.canonical_etypes:
    #     if etype != 'adj':
    #         continue
    #     # for i in range(0, g.num_edges((srctype,'adj',dsttype)), 10):
    #     agent = hg.find_edges(torch.arange(0, min(8, g.num_edges((srctype, 'adj', dsttype)))),
    #                           (srctype, 'adj', dsttype))
    #     plt.imshow(pic_b.items()[0])
    #     for i in range(len(agent[0])):
    #         src_data = hg.nodes[srctype].data['x'][agent[0][i]].view(8, 2)
    #         plt.scatter(src_data[:, 0], src_data[:, 1], c='b', marker='o')
    #         plt.text(src_data[0, 0], src_data[0, 1], "s {} {}".format(srctype, i), fontsize=7, color='black')
    #         src_data = hg.nodes[srctype].data['y'][agent[0][i]]
    #         plt.scatter(src_data[:, 0], src_data[:, 1], c='r', marker='x')
    #         src_data = predicted_future[srctype][agent[0][i],:,:]
    #         plt.scatter(src_data[:,0], src_data[:,1],c='g', marker='v')
    #
    #         dst_data = hg.nodes[dsttype].data['x'][agent[1][i]].view(8, 2)
    #         plt.scatter(dst_data[:, 0], dst_data[:, 1], c=colors[i], marker='o')
    #         plt.text(dst_data[0, 0], dst_data[0, 1], "d {} {}".format(dsttype, i), fontsize=7, color='black')
    #         dst_data = hg.nodes[dsttype].data['y'][agent[1][i]]
    #         plt.scatter(dst_data[:, 0], dst_data[:, 1], c=colors[i], marker='x')
    #         dst_data = predicted_future[dsttype][agent[1][i], :, :]
    #         plt.scatter(dst_data[:, 0], dst_data[:, 1], c='g', marker='v')

        plt.savefig(save_dir_base + '/' + scene + '_' + human + '.png', bbox_inches='tight', dpi=900)
        plt.close()



def plot_in_out_degree_distributions(edge_index, num_of_nodes, dataset_name):
    """
        Note: It would be easy to do various kinds of powerful network analysis using igraph/networkx, etc.
        I chose to explicitly calculate only the node degree statistics here, but you can go much further if needed and
        calculate the graph diameter, number of triangles and many other concepts from the network analysis field.
    """
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'
    if edge_index.shape[0] == edge_index.shape[1]:
        edge_index = convert_adj_to_edge_index(edge_index)

    # Store each node's input and output degree (they're the same for undirected graphs such as Cora)
    in_degrees = np.zeros(num_of_nodes, dtype=np.int)
    out_degrees = np.zeros(num_of_nodes, dtype=np.int)

    # Edge index shape = (2, E), the first row contains the source nodes, the second one target/sink nodes
    # Note on terminology: source nodes point to target/sink nodes
    num_of_edges = edge_index.shape[1]
    for cnt in range(num_of_edges):
        source_node_id = edge_index[0, cnt]
        target_node_id = edge_index[1, cnt]

        out_degrees[source_node_id] += 1  # source node points towards some other node -> increment it's out degree
        in_degrees[target_node_id] += 1  # similarly here

    hist = np.zeros(np.max(out_degrees) + 1)
    for out_degree in out_degrees:
        hist[out_degree] += 1

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.6)

    plt.subplot(311)
    plt.plot(in_degrees, color='red')
    plt.xlabel('node id'); plt.ylabel('in-degree count'); plt.title('Input degree for different node ids')

    plt.subplot(312)
    plt.plot(out_degrees, color='green')
    plt.xlabel('node id'); plt.ylabel('out-degree count'); plt.title('Out degree for different node ids')

    plt.subplot(313)
    plt.plot(hist, color='blue')
    plt.xlabel('node degree'); plt.ylabel('# nodes for a given out-degree'); plt.title(f'Node out-degree distribution for {dataset_name} dataset')
    plt.xticks(np.arange(0, len(hist), 5.0))

    plt.grid(True)
    plt.show()


def visualize_graph(edge_index, node_labels, dataset_name, visualization_tool=GraphVisualizationTool.IGRAPH):
    """
    Check out this blog for available graph visualization tools:
        https://towardsdatascience.com/large-graph-visualization-tools-and-approaches-2b8758a1cd59
    Basically depending on how big your graph is there may be better drawing tools than igraph.
    Note:
    There are also some nice browser-based tools to visualize graphs like this one:
        http://networkrepository.com/graphvis.php?d=./data/gsm50/labeled/cora.edges
    Nonetheless tools like igraph can be useful for quick visualization directly from Python
    """
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'
    if edge_index.shape[0] == edge_index.shape[1]:
        edge_index = convert_adj_to_edge_index(edge_index)

    num_of_nodes = len(node_labels)
    edge_index_tuples = list(zip(edge_index[0, :], edge_index[1, :]))

    # Networkx package is primarily used for network analysis, graph visualization was an afterthought in the design
    # of the package - but nonetheless you'll see it used for graph drawing as well
    if visualization_tool == GraphVisualizationTool.NETWORKX:
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(edge_index_tuples)
        nx.draw_networkx(nx_graph)
        plt.show()

    elif visualization_tool == GraphVisualizationTool.IGRAPH:
        # Construct the igraph graph
        ig_graph = ig.Graph()
        ig_graph.add_vertices(num_of_nodes)
        ig_graph.add_edges(edge_index_tuples)

        # Prepare the visualization settings dictionary
        visual_style = {}

        # Defines the size of the plot and margins
        visual_style["bbox"] = (3000, 3000)
        visual_style["margin"] = 35

        # I've chosen the edge thickness such that it's proportional to the number of shortest paths (geodesics)
        # that go through a certain edge in our graph (edge_betweenness function, a simple ad hoc heuristic)

        # line1: I use log otherwise some edges will be too thick and others not visible at all
        # edge_betweeness returns < 1.0 for certain edges that's why I use clip as log would be negative for those edges
        # line2: Normalize so that the thickest edge is 1 otherwise edges appear too thick on the chart
        # line3: The idea here is to make the strongest edge stay stronger than others, 6 just worked, don't dwell on it

        edge_weights_raw = np.clip(np.log(np.asarray(ig_graph.edge_betweenness()) + 1e-16), a_min=0, a_max=None)
        edge_weights_raw_normalized = edge_weights_raw / np.max(edge_weights_raw)
        edge_weights = [w**6 for w in edge_weights_raw_normalized]
        visual_style["edge_width"] = edge_weights

        # A simple heuristic for vertex size. Size ~ (degree / 2) (it gave nice results I tried log and sqrt as well)
        visual_style["vertex_size"] = [deg / 2 for deg in ig_graph.degree()]

        # This is the only part that's Cora specific as Cora has 7 labels
        if dataset_name.lower() == DatasetType.CORA.name.lower():
            visual_style["vertex_color"] = [cora_label_to_color_map[label] for label in node_labels]
        else:
            print('Feel free to add custom color scheme for your specific dataset. Using igraph default coloring.')

        # Set the layout - the way the graph is presented on a 2D chart. Graph drawing is a subfield for itself!
        # I used "Kamada Kawai" a force-directed method, this family of methods are based on physical system simulation.
        # (layout_drl also gave nice results for Cora)
        visual_style["layout"] = ig_graph.layout_kamada_kawai()

        print('Plotting results ... (it may take couple of seconds).')
        ig.plot(ig_graph, **visual_style)
    else:
        raise Exception(f'Visualization tool {visualization_tool.name} not supported.')


def draw_entropy_histogram(entropy_array, title, color='blue', uniform_distribution=False, num_bins=30):
    max_value = np.max(entropy_array)
    bar_width = (max_value / num_bins) * (1.0 if uniform_distribution else 0.75)
    histogram_values, histogram_bins = np.histogram(entropy_array, bins=num_bins, range=(0.0, max_value))

    plt.bar(histogram_bins[:num_bins], histogram_values[:num_bins], width=bar_width, color=color)
    plt.xlabel(f'entropy bins')
    plt.ylabel(f'# of node neighborhoods')
    plt.title(title)