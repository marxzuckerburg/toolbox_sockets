from graph_tool.all import *
from app_config import *
import time
from collections import defaultdict
import numpy as np,os
import pandas as pd

def ld2dld(ld,key):
	dld=defaultdict(list)
	for d in ld: dld[ d[key] ].append(d)
	return dld


def sims2net(most_similar_data,combine_periods=DEFAULT_COMBINED_PERIODS):
	nets=[]

	# print('combine_periods -->',combine_periods)
	
	if combine_periods=='diachronic':
		#print('msd by period!?',most_similar_data[0])
		df=pd.DataFrame(most_similar_data)
		for pi,period in enumerate(sorted(list(set(df.period)))):
			period_df=df[(df.period==period) & (df.period2==period)]
			period_data = period_df.to_dict('records')
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

		if np.isnan(csim): continue

		maybe_new_nodes = [[word1,node1], [word2,node2] ]
		
		for _word,_node in maybe_new_nodes:
			if _word not in nodes_sofar:
				nodes_sofar.append(_word)
				nodes.append(_node)
			
		# print(word1,'--',csim,'-->',word2)
		links.append({'source':word1, 'target':word2, 'weight':csim, 'was_path':data['was_path']})
	return {'nodes':nodes, 'links':links}





######
from collections import defaultdict

class GraphToolDB(object):

	def __init__(self,fn,log=None):
		self.fn=fn
		self.gt=None
		self.gt_orig=None
		self.n2v={}
		self.v2n={}
		self.nn2d=defaultdict(dict)
		self.log=log if log is not None else print

		self.filters={}
		self.ntop2gt={}
		self.edge_props=[]


	# def load_alts(self):
	# 	now=time.time()
	# 	print('loading alt graphs for: '+self.fn)
	# 	fndir=os.path.dirname(self.fn)
	# 	fnprefix=os.path.splitext(self.fn)[0]
	# 	for fn in os.listdir(fndir):
	# 		if fn.startswith('distnet.') and fn.endswith('.gt'):
	# 			fnfn=os.path.join(fndir,fn)
	# 			stamp = fn.split('.')[-2]
	# 			if stamp.startswith('n_top'):
	# 				n_top=int(stamp.split('=')[-1])
	# 				self.ntop2gt[n_top]=load_graph(fnfn,fmt='gt')
	# 	tdist=round(time.time()-now,2)
	# 	print('finished loading graph in %ss' % tdist)


	def load(self):
		print('loading graph: '+self.fn)
		now=time.time()
		g=self.gt=load_graph(self.fn,fmt='gt')
		tdist=round(time.time()-now,2)
		print('finished loading graph in %ss' % tdist)

		# init
		for v in g.vertices():
			v_id=g.vp.id[v]
			self.n2v[v_id]=v
			self.v2n[v]=v_id

		self.edge_props = list(self.gt.edge_properties.keys())
		# for a,b,d,e in self.edges():
		# 	self.nn2d[a][b]=d

		#self.load_alts()

	def edges(self,data=True):
		for e in self.gt.edges():
			v1,v2=e.source(),e.target()
			n1,n2=self.v2n[v1],self.v2n[v2]
			# props
			d={}
			for k in self.gt.edge_properties.keys():
				d[k]=self.gt.edge_properties[k][e]

			yield (n1,n2,d,e)

	def neighbors(self,n):
		return [self.v2n[v] for v in self.n2v[n].out_neighbors()]

	def most_similar_data(self,n,n_top=None):
		if not n in self.n2v: return []
		v1=self.n2v[n]
		odat=[]
		for v2 in v1.out_neighbors():
			n2=self.v2n[v2]
			e=self.gt.edge(v1,v2)

			odx={
			# 'source':n, 'target':n2
			'word':n2
			}
			for eprop in self.edge_props:
				odx[eprop]=self.gt.ep[eprop][e]
			odat+=[odx]
		return odat


	def filter_edges(self,prop_name,min_prop_val=None,max_prop_val=None):
		#self.gt.set_edge_filter(None)

		self.log(f'filtering edges for {prop_name} ({min_prop_val} < x < {max_prop_val}) ...')
		now=time.time()

		eprop_id=(prop_name,min_prop_val,max_prop_val)
		if eprop_id in self.filters:
			return self.filters[eprop_id]
		else:			
			
			now=time.time()
			eprop_filter=self.gt.new_edge_property('bool')
			for a,b,d,e in self.edges():
				use_edge=True
				prop_val=d.get(prop_name)
				if not prop_val: use_edge=False
				if min_prop_val and prop_val<min_prop_val: use_edge=False
				if max_prop_val and prop_val>max_prop_val: use_edge=False
				eprop_filter[e]=use_edge
		
		#self.gt.set_edge_filter(eprop_filter)

		tdist=round(time.time()-now,1)
		self.log('>> edges filtered in %ss' % tdist)
		gq = self.filters[eprop_id] = GraphView(self.gt, directed=True, efilt=eprop_filter)
		return gq

	def filter_rank(self,max_rank=10):
		# hack
		if max_rank==50: return self.gt

		if max_rank in self.ntop2gt:
			return self.ntop2gt[max_rank]

		return self.filter_edges('sim_rank',max_prop_val=max_rank)
		#return self.gt

	def shortest_path(self,n1,n2,weight_prop='cdist',g=None):
		g=self.gt if g is None else g
		weights = g.edge_properties[weight_prop] if weight_prop in g.edge_properties else None
		try:
			vlist,elist = shortest_path(g,self.n2v[n1],self.n2v[n2],weights=weights)
			npath = [self.v2n[v] for v in vlist]
			return npath
		except KeyError:
			return None




if __name__ == '__main__':
	G=GraphToolDB(fn='models/COHA_byhalfcentury_nonf_smpl/distnet.gt')
	G.load()
	print(G.n2v)

	for a,b,d in G.edges(data=True):
		print(a,b,d)















"""
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

"""