import pandas as pd
import numpy as np
import math
import operator
import copy


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
    while (remaining_mempool_data.shape[0])>0 and z >-1:
        gt_tracker+=1
        # get next block of transitions
        transitions_in_block = greedy_knapsack(block_size, remaining_mempool_data)

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
