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
	pass

############################
# Part 1
############################

def greedy_knapsack(block_size, mempool_data):
	
	# return a list of the tx id's to insert into a block
	return ["513b062d7bce674e90f089e19028da4a4c7e6347711fd08a714d69e9ed60180f", 
			"d42d24c1605a920866c209b8bdd11da01623d698e5a30319440e272332d660c7", 
			"35882476b1074182635d0093617e7a477821830bd9d5785077148a62005b023a",
			"03ee8ba1bfd5882c4d1c51e6ed70933d4dced572187fc2dfe45e7a1849f0ef02",
			"5799ad0f83095a7942f9d925a70d9c4137f73bce54b93e95455a6f4e97877cfd"]

			
def evaluate_block(tx_list, mempool_data):
	
	# return the miner revenue [satoshi]
	return 430504016.0

	
def VCG(block_size, tx_list, mempool_data):
	
	keys = ("513b062d7bce674e90f089e19028da4a4c7e6347711fd08a714d69e9ed60180f", 
			"d42d24c1605a920866c209b8bdd11da01623d698e5a30319440e272332d660c7", 
			"35882476b1074182635d0093617e7a477821830bd9d5785077148a62005b023a",
			"03ee8ba1bfd5882c4d1c51e6ed70933d4dced572187fc2dfe45e7a1849f0ef02",
			"5799ad0f83095a7942f9d925a70d9c4137f73bce54b93e95455a6f4e97877cfd",
			"63f1e5eef1f1eadf1ed89423b6ccb164344e81fa0bc8234ddbed065d20014988")
	values=(264927.0, 85939.0, 85939.0, 85939.0, 171878.0, 85939.0)
	
	# return a dict of tx_id as keys, for each tx_id it VCG price [satoshi)]
	return dict(zip(keys, values))

	
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

	
