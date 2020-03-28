from app_config import *

from collections import defaultdict

def ld2dld(ld,key):
	dld=defaultdict(list)
	for d in ld: dld[ d[key] ].append(d)
	return dld


def sims2net(most_similar_data,combine_periods=DEFAULT_COMBINED_PERIODS):
	nets=[]

	#print('combine_periods -->',combine_periods)
	
	if combine_periods=='diachronic':
		#print('msd by period!?',most_similar_data[0])
		most_similar_data_by_period = ld2dld(most_similar_data,'period')
		for period in most_similar_data_by_period:
			period_data = most_similar_data_by_period[period]
			#print('period!!',period,period_data)
			net = mostsim2netjson(period_data)
			netd = {'period':period, 'netdata':net}
			nets.append(netd)
	else:
		net = mostsim2netjson(most_similar_data)
		netd = {'period':None, 'netdata':net}
		nets.append(netd)

	#print('NETSSS:',nets)

	return nets



def mostsim2netjson(most_similar_data):
	nodes = []
	nodes_sofar = []
	links = []

	for i,data in enumerate(most_similar_data):
		#print(i,data)
		node1={'id':data['id'], 'word':data['word'], 'period':data['period']}
		node2={'id':data['id2'], 'word':data['word2'], 'period':data['period2']}
		word1=data['id']
		word2=data['id2']
		csim=data['csim']

		maybe_new_nodes = [[word1,node1], [word2,node2] ]
		
		for _word,_node in maybe_new_nodes:
			if _word not in nodes_sofar:
				nodes_sofar.append(_word)
				nodes.append(_node)
			
		#print(word1,'--',csim,'-->',word2)
		links.append({'source':word1, 'target':word2, 'weight':csim})
	return {'nodes':nodes, 'links':links}

