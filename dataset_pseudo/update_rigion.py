import numpy as np
import os
import pickle
import json

if __name__ == '__main__':
    file_List = os.listdir()
    region_list = []
    for file in file_List:
        if os.path.isdir(file):
            region_list.append(file)
    print(region_list)

    pair_list = {}
    
    pair_list['region'] = region_list
    Num_Pair = 0
    for R in region_list:
        pair_list[R] = {}
        try:
            R_pair_path = os.path.join(R, 'pair_data', 'relation.npy')
            R_pair = np.load(R_pair_path)
            pair_idx = np.argwhere(R_pair>0)
            Num_Pair += len(pair_idx)
            """
            pair idx에 따라 순서가 바뀔수 있으므로 절대적인 순서를 저장하도록 변경
            """
            pair_list[R]["N"] = Num_Pair
        except:
            print("No relation.npy ")
        
    pair_list['Num_Pair'] = Num_Pair
    """
    TODO
    """




    with open('pair_list.json', 'w') as outfile:
        json.dump(pair_list, outfile)
        

