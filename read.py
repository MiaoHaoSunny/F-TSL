import pandas as np
import pandas as pd
import collections




# data_1 = pd.read_csv('./road_network/rome_taxi_1/Edge.csv',sep=',')
# data_2 = pd.read_csv('./data/rome_taxi_1/road/edge_weight.csv',sep=',')
# data_3 = pd.read_csv('./data/rome/road/edge.csv',sep=',')
#
# merge = pd.merge(data_1,data_2,how='inner',on=['s_node','e_node'])
# merge_2 = pd.merge(data_3, merge, how='inner', on=['s_lng', 's_lat', 'e_lng', 'e_lat'])
# df = pd.DataFrame({'section_id':merge_2.section_id,'s_node':merge_2.s_node_x,'e_node':merge_2.e_node_x,'length':merge_2.length})
# # merge.to_csv('./data/rome_taxi_1/road/edge.csv',sep=',',header=True, index = False)
# df.to_csv('./data/rome_taxi_1/road/edge_weight.csv',sep=',',header=True, index = False)


for i in range(1,26):
    data_1 = pd.read_csv('./data/rome/road/node.csv',sep=',')
    data_2 = pd.read_csv('./road_network/rome_taxi_{}/Point.csv'.format(i), sep=',')
    merge = pd.merge(data_1,data_2,how='inner',on=['lng','lat'])
    df = pd.DataFrame({"node":merge.node_x,"lng":merge.lng,"lat":merge.lat})
    df.to_csv('./road_network/rome_taxi_{}/match.csv'.format(i),sep=',',header=True, index = False)

    traj = pd.read_csv('./road_network/rome_taxi_{}/match.csv'.format(i),sep=',')
    matching_result = pd.read_csv('./data/rome/st_traj/matching_result.csv')
    node_list = matching_result.Node_list
    traj_id = pd.DataFrame({'node':traj.node})
    node_list_int = []
    b = traj_id['node'].to_list()
    for nlist in node_list:
        tmp_list = []
        nlist = nlist[1:-1].replace('[', '').replace(']','' ).replace(' ', ',').replace('\n', ',').split(',')
        for n in nlist:
            if n != '':
                tmp_list.append(int(n))
        # node_list_int.append(tmp_list)
        node_df = pd.DataFrame({'node':tmp_list})
        a = node_df['node'].to_list()
        match = [x for x in a if x in b]
        if(collections.Counter(match) == collections.Counter(a)):
            node_list_int.append(tmp_list)
    node_list_int = np.array(node_list_int)
    df = pd.DataFrame({'Node_list':node_list_int})
    df.to_csv('./road_network/rome_taxi_{}/matching_node.csv'.format(i),sep=',',header=True, index = False)
    print("finish taxi id",i)
