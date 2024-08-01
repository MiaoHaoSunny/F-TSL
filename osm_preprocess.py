import random
import numpy as np
import pickle
import pandas as pd
import osm2gmns as og
import osm2gmns.settings as defaultsettings
import yaml


def gen_network_data(i):
    net = og.getNetFromOSMFile('./road_network/{}_taxi_{}/{}_taxi_{}.osm'.format(dataset, i, dataset, i), POIs=False, default_lanes=True, default_speed=True)
    og.outputNetToCSV(net, './road_network/{}_taxi_{}'.format(dataset,i))

    df = pd.read_csv('./road_network/{}_taxi_{}/node.csv'.format(dataset,i),header=0,usecols=['node_id','x_coord','y_coord'])
    df.columns = ['node', 'lng', 'lat']
    df.to_csv('./road_network/{}_taxi_{}/Point.csv'.format(dataset,i), index = False)

    df = pd.read_csv('./road_network/{}_taxi_{}/link.csv'.format(dataset,i),header=0,usecols=['link_id','from_node_id','to_node_id','length'])
    df.columns = ['section_id', 's_node', 'e_node', 'length']
    df.to_csv('./road_network/{}_taxi_{}/edge_weight.csv'.format(dataset,i), index = False)

    df = pd.read_csv('./road_network/{}_taxi_{}/link.csv'.format(dataset,i),usecols=['link_id','from_node_id','to_node_id','geometry'])
    coors = df.geometry
    s_lngs = []
    s_lats = []
    e_lngs = []
    e_lats = []
    c_lngs = []
    c_lats = []
    for coor in coors:
        coor = coor.split('(')[-1].split(')')[0]

        s = coor.split(', ')[0]
        s_lng, s_lat = s.split(' ')

        e = coor.split(', ')[-1]
        e_lng, e_lat = e.split(' ')

        s_lngs.append(s_lng)
        s_lats.append(s_lat)
        e_lngs.append(e_lng)
        e_lats.append(e_lat)
        c_lngs.append((float(s_lng)+float(e_lng))/2)
        c_lats.append((float(s_lat)+float(e_lat))/2)

    ddf = pd.DataFrame({"edge": df.link_id, "s_node": df.from_node_id, "e_node": df.to_node_id, "s_lng": s_lngs, "s_lat": s_lats, "e_lng": e_lngs, "e_lat": e_lats, "c_lng": c_lngs, "c_lat": c_lats})
    ddf.to_csv('./road_network/{}_taxi_{}/Edge.csv'.format(dataset,i) ,sep=',',header=True, index = False)

if __name__ == '__main__':
    # config = yaml.safe_load(open('config.yaml'))
    config = yaml.safe_load(open('config_rome.yaml'))
    dataset = config["dataset"]
    for i in range(15,26):
        gen_network_data(i)
        print("finish taxi id:",i)

