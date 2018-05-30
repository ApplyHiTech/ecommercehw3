import pandas as pd
import numpy as np
import math
import operator
import copy
import os


def load_mempool_data(mempool_data_full_path, current_time=1510264253.0):
    #read json to pandas
    df = pd.read_json(mempool_data_full_path,orient='index')

    # filter for time(T)<current_time < Remove(T)
    df_before = df['time']<current_time
    df_after = df['removed']>=current_time

    df_pending = df[df_before & df_after]

    # return pending transactions
    # test
    return df_pending


############################
# Part 1
############################

def greedy_knapsack(block_size, mempool_data):

    mempool_data['txid'] = mempool_data.index
    # Add column to DF mempool_data to compute fee_per_size = (fee / size)

    mempool_data['fee_per_size'] = mempool_data['fee']/ mempool_data['size']
    # Order the df in descending fee_per_size, id asc

    mempool_data_sorted=mempool_data.sort_values(by=['fee_per_size','txid'],ascending=[False,True])

    #implement Greedy knapsack.
    list_tx = [] # return the list of transaction ids selected
    block_size_remaining = block_size
    #optimization
    min_size = mempool_data_sorted['size'].min()

    for index,row in mempool_data_sorted.iterrows():
        if row['size']<=block_size_remaining:
            #add transaction
            list_tx.append(row['txid'])
            block_size_remaining-=row['size']

        if block_size_remaining < min_size:
            break

    return list_tx

def evaluate_block(tx_list, mempool_data):
    val = 0
    # return the miner revenue [satoshi]

    for itx in tx_list:
        val+=mempool_data.loc[itx]['fee']

    return val

def VCG_tx(block_size,txid,tx_list,mempool_data):
    val = 0
    # Compute V^S_{b-txi}
    bl_size = mempool_data.loc[txid]['size']
    #print("txid: "+ str(txid)+ " " +str(bl_size) + " fee: " + str(mempool_data.loc[txid]['fee']))

    tran_list = greedy_knapsack(block_size+bl_size,mempool_data)
    v_s= evaluate_block(tran_list,mempool_data)-mempool_data.loc[txid]['fee']

    # Compute V^{S-j}_{b-txi}
    v_s_j = evaluate_block(tx_list,mempool_data)-mempool_data.loc[txid]['fee']
    #print("v_s: " + str(v_s)+ " v_s_j " + str(v_s_j) + " p: " +str(v_s-v_s_j) + "  vs " + str(mempool_data.loc[txid]['fee']))
    return v_s-v_s_j
    _
def VCG(block_size, tx_list, mempool_data):
    ret = {}
    for itx in tx_list:

        val = VCG_tx(block_size,str(itx),tx_list,mempool_data)
        if val < 0:
            val = 0
        ret[str(itx)]=val

    # return a dict of tx_id as keys, for each tx_id it VCG price [satoshi)]
    return ret

############################
# Part 2
############################

def utility(value,urgency,tx_size,z,gt_z):
    #if gt_z = never so it is undefined and is -1.
    if gt_z == -1:
        return 0

    exp_value = (-1 * gt_z * urgency / 1000)


    return  value*math.pow(2, exp_value) - z * tx_size

def forward_bidding_agent(tx_size, value, urgency, mempool_data, block_size):
    #print(mempool_data)

    #compute gt_z values.
    dict_gt= generate_gt(mempool_data,block_size,value)



    #compute utilitys
    dict_gu = {} # Dictionary key = z, value = gu(z)
    res = []
    for z in np.arange(0, 5001, 10):  # includes 5,000
        #print(z)
        dict_gu[z] = utility(value,urgency,tx_size,z,dict_gt[z])
        res+= [(z,dict_gt[z],dict_gu[z])]

    #rdf = pd.DataFrame(list(res))
    #print(rdf)
    return max(dict_gu.items(), key=operator.itemgetter(1))[0]


def truthful_bidding_agent(tx_size, value, urgency, mempool_data, block_size):



    z = value*(2**(-3.6*urgency))
    return z

def get_min_fee(block,mempool_data):
    min_fee_in_block = -1
    for t in block:
        temp = mempool_data.loc[t]['fee_per_size']
        if min_fee_in_block == -1:
            min_fee_in_block = temp
        else:
            min_fee_in_block = min(temp, min_fee_in_block)
    return min_fee_in_block


def generate_gt(mempool_data,block_size,value):
    # This is a function that guesses the time for a fee z and returns the time in seconds.
    # iteration 1.
    remaining_mempool_data = copy.deepcopy(mempool_data)
    gt_tracker = 0
    z = 5000
    dict_gt = {}
    flag = True
    while (remaining_mempool_data.shape[0])>0 and z >-1 and flag:
        gt_tracker+=1
        # get next block of transitions
        transitions_in_block = greedy_knapsack(block_size, remaining_mempool_data)
        if len(transitions_in_block) == 0:
            flag = False

        # get min fee per size
        min_fee_in_block = get_min_fee(transitions_in_block,mempool_data)
        time = gt_tracker*60

        while z >= min_fee_in_block:

            dict_gt[z]=time
            z-=10

        # remove transitions from remaining_mempool_data
        for txid in transitions_in_block:

            remaining_mempool_data = remaining_mempool_data[remaining_mempool_data.txid != txid]

    while z > -1:
        # set to 'never' which is -1
        dict_gt[z]=-1
        z-=10

    return dict_gt
"""
def new_bidding_agent(value,urgency,size):
    #default to ti = 750
    if value/size < 500:
        # we don't want to bid on this! Because it will be added late
        # and will deliver super low value to us.
        return 0

    elif urgency > 0.9:
        return 0

    elif urgency < 0.01:
        return 0

    else:

        return 500

    #3.5M
    #return 500

    # 753K
    if urgency < 0.5:
    
    
    
        return 0
    else:

        z = (value/size)*urgency
    return  z
   
def update_z_values_of_agent(df):
    # iterate through df
    #print("update z")

    for index,row in df.iterrows():
        value = row["v"]
        urgency = row["r"]
        size = row["size"]
        df.at[index,"z"] = new_bidding_agent(value,urgency,size)
    # set z = new_bidding_agent(value,urgency,size) for each row

    return df

def evaluate_competition(mempool_data,block_size=1000):

    # Read the hw2_part2 csv file
    df_part2 = pd.read_csv(os.path.abspath("hw2_part2.csv"))

    #print(df_part2)
    df_part2 = update_z_values_of_agent(df_part2)
    #print(df_part2)
    z_values= df_part2
    #z_index = df_part2.set_index("z")

    # dict_tx_z = dictionary of key = tx_id and value = z (bid)
    # run greedy on the mempool_data and output # dictionary of tx_id: time, fee_per_byte, time_o, time_initial
    # tx_id, fee_per_byte, time_o, time_initial
    remaining_mempool_data = copy.deepcopy(mempool_data)
    gt_tracker = 0
    dict_tx_id_time = {}
    j = 1
    flag = True
    while (remaining_mempool_data.shape[0]) > 50 and flag:
        if remaining_mempool_data.shape[0]%500 ==0:
            print(remaining_mempool_data.shape[0])
        j+=1
        # print(j)
        gt_tracker += 1
        # get next block of transitions
        transitions_in_block = greedy_knapsack(block_size, remaining_mempool_data)
        if len(transitions_in_block) == 0:
            flag = False

        # get min fee per size
        #time = gt_tracker * 60 + 1510264253.0

        # remove transitions from remaining_mempool_data
        for txid in transitions_in_block:
            data = mempool_data.loc[txid]
            remove_time = data['removed']
            enter_time = data['time']
            t_diff = remove_time-enter_time
            #print("Enter %s , Remove %s, Diff %s" % (enter_time,remove_time,t_diff))
            dict_tx_id_time[txid]={"time":remove_time,"fee_per_size":data['fee']/ data['size'],"time_initial":data['time'],"t_diff":t_diff}
            remaining_mempool_data = remaining_mempool_data[remaining_mempool_data.txid != txid]
    print(dict_tx_id_time)

    # compute l2 value between (dict_tx_z and mempool_data 'fee-per-byte' for all transactions
    #z_values = list(df_part2["Z"])

    # THE TRANSACTION IDS ARE ALL RETURNED TEH SAME BUT I NEED TO GO TO SLEEP.
    min_pairs_tuple = compute_differences(z_values,dict_tx_id_time)

    #get_top_ten k with smallest l2 values.
    k = 10
    df_tup = pd.DataFrame(min_pairs_tuple, columns=["v","r","size","z","tx","min_diff","t_diff"]).sort_values(by=["t_diff"])
    #print(df_tup)
    df_tup_min_k = df_tup[:k]
    #print(df_tup[:k])
    #print("Hey")
    new_tup = []
    for index,row in df_tup_min_k.iterrows():
        tx = row["tx"]
        z = row["z"]
        v = row["v"]
        r = row["r"]
        size = row["size"]
        time_diff = row["t_diff"]
        min_diff = row["min_diff"]
        new_tup += [{"tx":tx,
                    "t_diff":time_diff,
                    "min_diff":min_diff,
                    "z":z,
                    "value": v,
                    "urgency":r,
                    "block_size":size}]
    print(new_tup)
    print(compute_all_w(new_tup))

def compute_all_w(list_of_info):
    #print(list_of_info)
    val = 0
    for i in list_of_info:
        #print(i)
        t_diff = i["t_diff"]
        urgency = i["urgency"]
        value = i["value"]
        z = i["z"]
        block_size = i["block_size"]
        print("value: %s * 2 ** (-%s*%s)-%s*%s" % (value,t_diff,urgency,z,block_size))
        val += compute_w(t_diff,urgency,value,z,block_size)
    return val

    # compute sum wi for all k transactions.
    # t_diff =0 #time difference
    # urgency = 0 #given as r
    # value = 0 #given as v
    # z = 0 #bid
    # block_size
def compute_w(t_diff,urgency,value,z,block_size):

    val = value*(2**(-t_diff*urgency/1000))-z*block_size
    print("w_i = %s" % val)
    return val

def compute_differences(z_values,dict_tx_id_time):
    #returns dictionary of closest tx to bid
    ret_dict = {}
    tup=[]
    #print("TEST")
    large_number = 1000000
    #print(z_values)
    for index,row in z_values.iterrows():
        z = row["z"]
        val= row["v"]
        r = row["r"]
        size = row["size"]
        min_diff = large_number
        tran_i = -1
        #print("Length of Dict_tx %s" % len(dict_tx_id_time))
        t_diff = -1
        for i in dict_tx_id_time:
            a = dict_tx_id_time[i]["fee_per_size"]
            b = z
            #print ("a = %s, b = %s" %(a,b))
            l2_val = l2_norm(a,b)

            if min_diff > l2_val:
                min_diff = l2_val
                tran_i = i
                t_diff = dict_tx_id_time[i]["t_diff"]

        #ret_dict[z] = {"v":row["v"],"tx": tran_i, "diff": min_diff}
        tup+=[(val,r,size,z,tran_i,min_diff,t_diff)]
    #print("End  of compute")
    #print(ret_dict)
    return tup

def l2_norm(a,b):
    #print("L2 Norm")
    x= math.pow(abs(math.pow(a,2)-math.pow(b,2)),.5)
    #print(x)
    return x
"""
