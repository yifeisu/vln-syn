import sys
import json
import math
import random
from tqdm import tqdm

import MatterSim
import networkx as nx
import numpy as np

from r2r_dataset import read_img_features

sys.path.append('E:/4-MyResearch_Task/0-vln/3-vln_space/pretrain_src/utils/')
sys.path.append('E:/4-MyResearch_Task/0-vln/3-vln_space/finetune_src/')


def load_nav_graphs(_scans):
    """ Load connectivity graph for each scan """

    def distance(pose1, pose2):
        """ Euclidean distance between two graph poses """
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    for scan in _scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def _load_nav_graphs(scans):
    """ Load connectivity graph for each scan, useful for reasoning about shortest paths """
    graphs = load_nav_graphs(scans)
    paths = {}
    for scan, G in graphs.items():  # compute all shortest paths
        paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
    distances = {}
    for scan, G in graphs.items():  # compute all shortest paths
        distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    return paths, graphs


def _shortest_path_action(scan, current_viewpoint, goalviewpoint, _paths):
    """ Determine next action on the shortest path to goal, for supervised training. """
    if current_viewpoint == goalviewpoint:
        return goalviewpoint  # do nothing

    path = _paths[scan][current_viewpoint][goalviewpoint]
    next_viewpoint_id = path[1]

    return next_viewpoint_id


def make_candidate(_sim, scanId, viewpointId, viewId):
    def _loc_distance(_loc):
        return np.sqrt(_loc.rel_heading ** 2 + _loc.rel_elevation ** 2)

    base_heading = (viewId % 12) * math.radians(30)
    adj_dict = {}
    long_id = "%s_%s" % (scanId, viewpointId)
    for ix in range(36):
        if ix == 0:
            _sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            _sim.makeAction([0], [1.0], [1.0])
        else:
            _sim.makeAction([0], [1.0], [0])

        state = _sim.getState()[0]
        assert state.viewIndex == ix

        # Heading and elevation for the viewpoint center
        heading = state.heading - base_heading
        elevation = state.elevation

        # get adjacent locations
        for j, loc in enumerate(state.navigableLocations[1:]):
            # if a loc is visible from multiple view, use the closest view (in angular distance) as its representation
            distance = _loc_distance(loc)
            # heading and elevation for the loc
            loc_heading = heading + loc.rel_heading
            loc_elevation = elevation + loc.rel_elevation
            # angle_feat = angle_feature(loc_heading, loc_elevation)
            if loc.viewpointId not in adj_dict or distance < adj_dict[loc.viewpointId]['distance']:
                adj_dict[loc.viewpointId] = {'scanId': scanId,
                                             'viewpointId': loc.viewpointId,  # Next viewpoint id
                                             'pointId': ix,
                                             'distance': distance,
                                             'loc_rela_angle': [loc_heading, loc_elevation]}
    candidates = list(adj_dict.values())
    return candidates


if __name__ == '__main__':
    file2generate = 'prevalent_aug.json'

    # -------------------------------------------------------------------------------------- #
    # read the image features and initialize the simulator
    # -------------------------------------------------------------------------------------- #
    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(60))
    sim.setBatchSize(1)
    sim.initialize()

    # -------------------------------------------------------------------------------------- #
    # load the connectivty and build the graph
    # -------------------------------------------------------------------------------------- #
    with open(r"connectivity\scans.txt", 'r') as f:
        scans = f.readlines()
    scans = [scan.strip() for scan in scans]

    paths, graphs = _load_nav_graphs(scans)
    print('finish loading the graphs.')

    # -------------------------------------------------------------------------------------- #
    # read the prevalent json file and create the new json data file
    # -------------------------------------------------------------------------------------- #
    with open('prevalent_pretrain/shortest_%s' % file2generate, 'r') as f:
        data = json.load(f)
    print('finish loading the prevalent json file.')

    new_data = list()
    bar = tqdm(data, desc='progress', total=len(data), ncols=200, colour='blue')
    for index, item in enumerate(bar):
        new_item = dict()
        new_item['instr_ids'] = item['instr_encoding']
        new_item['path'] = item['path']
        new_item['traj_scan'] = item['path'][0][0]

        # 1. generate the candidate views for each viewpoint in the path
        cands_view_index = list()
        cands_rela_angle = list()
        next_viewpointids = list()
        next_views_longid = list()
        for t, point in enumerate(new_item['path']):
            # a list of dicts, each dict records a information of candidate
            candidates = make_candidate(sim, point[0], point[1], point[2])
            cands_view_index.append([candidate['pointId'] for candidate in candidates])
            cands_rela_angle.append([candidate['loc_rela_angle'] for candidate in candidates])

            next_views_longid = [candidate['viewpointId'] for candidate in candidates]
            next_view = _shortest_path_action(point[0], point[1], new_item['path'][-1][1], paths)
            if point[1] != new_item['path'][-1][1]:
                next_viewpointids.append(next_views_longid.index(next_view))
            else:
                next_viewpointids.append(-1)

        # 1. random select a viewpoint in path and create the candidate views
        # random_viewpoint = random.choice(new_item['path'][:-1])
        # new_item['random_viewpoint4nap'] = random_viewpoint
        # candidates = make_candidate(sim, random_viewpoint[0], random_viewpoint[1], random_viewpoint[2])
        # cand_long_id = list()
        # cand_view_idex = list()
        # cand_rela_angle = list()
        # for candidate in candidates:
        #     cand_long_id.append('%s_%s' % (candidate['scanId'], candidate['viewpointId']))
        #     cand_view_idex.append(candidate['pointId'])
        #     cand_rela_angle.append(candidate['loc_rela_angle'])

        new_item['cands_view_index'] = cands_view_index
        new_item['cands_rela_angle'] = cands_rela_angle
        new_item['next_viewpointids'] = next_viewpointids

        new_data.append(new_item)
        bar.set_postfix_str(f"Processing the {index+1} item.")

    # -------------------------------------------------------------------------------------- #
    # write the new generated json data file
    # -------------------------------------------------------------------------------------- #
    with open(file2generate, 'w') as f:
        json.dump(new_data, f, sort_keys=False)
    print('finish writing our new json file.')
