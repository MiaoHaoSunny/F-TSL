import pandas as pd
import os

for i in range(1,26):
    data_traj = pd.read_csv('./data/rome/st_traj/matching_result.csv',sep=',')
    data_time = pd.read_csv('./data/rome/st_traj/time_drop_list.csv',sep=',')
    data_match = pd.read_csv('./road_network/rome_taxi_{}/matching_node.csv'.format(i),sep=',')

    merge_1 = pd.merge(data_traj,data_match,how='inner',on=['Node_list'])
    os.makedirs('./data/rome_taxi_{}/st_traj'.format(i), exist_ok=True)
    merge_1.to_csv('./data/rome_taxi_{}/st_traj/matching_result.csv'.format(i),sep=',',header=True,index=False)
    merge_2 = pd.merge(merge_1,data_time, how='inner',on='Traj_id')
    merge_2 = pd.DataFrame({'Traj_id':merge_2.Traj_id,'Time_list':merge_2.Time_list})
    merge_2.to_csv('./data/rome_taxi_{}/st_traj/time_drop_list.csv'.format(i),sep=',',header=True, index=False)
    print("finish taxi id:", i)
# merge_2 = pd.merge(data_time,data_match,how='inner',on=['Node_List'])