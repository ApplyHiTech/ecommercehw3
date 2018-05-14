'''
you should fill the func given here
all the other imports/constants/classes/func should be stored here and only here (not in other files)
'''

############################
# Insert your imports here
############################
import pandas as pd



def load_mempool_data(mempool_data_full_path, current_time=1510264253.0):
	#read json to pandas
	df = pd.read_json(mempool_data_full_path,orient='index')

	# filter for time(T)<current_time < Remove(T)
	df_before = df['time']<current_time
	df_after = df['removed']>=current_time

	df_pending = df[df_before & df_after]

	# return pending transactions
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
		ret[str(itx)]=val

	# return a dict of tx_id as keys, for each tx_id it VCG price [satoshi)]
	return ret

############################
# Part 2
############################
	
def forward_bidding_agent(tx_size, value, urgency, mempool_data, block_size):

	z = 100.00
	# return the bidder bid [satoshi]
	return z
	

def truthful_bidding_agent(tx_size, value, urgency, mempool_data, block_size):

	z = value*(2**(-3.6*urgency))
	return z

	
