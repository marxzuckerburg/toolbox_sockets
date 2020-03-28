import logging,time,os
import numpy as np
import pandas as pd
import db as DB
from tqdm import tqdm
from app_config import *
from tinydb import Query
from graph_tool.all import *

#  global variables for storage
fn2M = {}
fn2voc={} 





class Embedding(object): 

	def __init__(self,id,fn,periods=[]):
		self.fn = fn
		self.periods=periods
		self.id=id
		self.log=lambda x: print(x+'\n') #logging.info
		self.progress=logging.info


	## Three methods of storage/calculation
	
	## 1) Gensim
	@property
	def gensim(self):
		import gensim
		if not hasattr(self,'_gensim'):
			now=time.time()
			#self.log('generating model from vectors stored in db: '+self.fn)
			# self._gensim = gensim.models.KeyedVectors(vector_size=300)
			# all_words,all_vecs = self.get_all_word_vecs()
			# self._gensim.add(all_words,all_vecs)

			self.log('generating model from vectors stored on disk: '+self.fn)
			all_words=[]
			all_vecs=[]
			with open(self.fn) as f:
				for i,ln in enumerate(f):
					if not i: continue
					lndat=ln.strip().split()
					all_words.append(lndat[0])
					all_vecs.append([float(x) for x in lndat[1:]])
			self._gensim = gensim.models.KeyedVectors(vector_size=300)
			self._gensim.add(all_words,all_vecs)

			# self._gensim=gensim.models.KeyedVectors.load_word2vec_format(self.fn)
			
			tdist=round(time.time()-now,1)
			self.log('done loading model in %ss' % tdist)
		return self._gensim

	## 2) VecDB
	@property
	def db(self):
		if not hasattr(self,'_db'): self._db=DB.get_vecdb(self.id)
		return self._db

	## 3) DistNet
	@property
	def distdb(self):
		if not hasattr(self,'_distdb'): self._distdb=DB.get_distdb(self.id)
		return self._distdb

	@property
	def distnet_fn(self):
		#if hasattr(self,'fn_distnet'): return self.fn_distnet
		#return os.path.splitext(self.fn)[0]+'.distnet.txt'
		return os.path.join(DB_DIR,self.id,'distnet.gt')


	@property
	def distnet(self):
		if not hasattr(self,'_distnet'): self._distnet=self.get_distnet()
		return self._distnet

	def get_distnet(self,n_top=None):
		from networks import GraphToolDB
		G=GraphToolDB(fn=self.distnet_fn,log=self.log)
		G.load()
		return G

		#self.log('building network...')
		# import networkx as nx
		# G=nx.DiGraph()
		# #for d in tqdm(self.distdb.find(),total=MAX_NUM_VECS_TO_STORE):
		# for d in self.distdb.all():
		# 	if n_top and d['sim_rank']>n_top: continue
		# 	G.add_edge(d['source'],d['target'],weight=d['weight'],sim_rank=d['sim_rank'])
		# return G
		#return self.distdb

	## DATA GENERATION

	# def build_vecdb(self,max_num_vecs=None):
	# 	with open(self.fn) as f:
	# 		for i,ln in enumerate(tqdm(f,total=None)):
	# 			if i==0: continue
	# 			# if i>max_num_vecs: break
	# 			lndat=ln.split()
	# 			word=lndat[0]
	# 			wordvecs=[float(x) for x in lndat[1:]]
	# 			self.db.insert({'word':word,'vecs':wordvecs})

	
	def build_distmatrix(self,n_top=DEFAULT_N_TOP,n_vecs=MAX_NUM_VECS_TO_STORE):
		self.log('BUILDING DISTANCE DATABASE')
		
		all_words,all_vecs=self.get_all_word_vecs()
		vecsM = np.array(all_vecs[:n_vecs])
		vocab=all_words[:n_vecs]
		print('vecs shape',vecsM.shape)

		# scikit?
		import time
		now=time.time()
		from sklearn.metrics import pairwise_distances
		arr = pairwise_distances(vecsM, metric='cosine',n_jobs=4)
		tdist=round(time.time()-now,1)
		print('done computing %s vecs in %ss' % (n_vecs,tdist))

		# pandas?
		now=time.time()
		import pandas as pd
		dfdist=pd.DataFrame(arr,index=vocab,columns=vocab)
		tdist=round(time.time()-now,1)
		print('done computing pandas in %ss' % tdist)
		# print(dfdist)

		# save?
		#ofn=os.path.join(DB_DIR,self.id,'distmatrix.csv')
		#dfdist.to_csv(ofn)
		# ofn=os.path.join(DB_DIR,self.id,'distdf.pickle')
		# dfdist.to_pickle(ofn)
		return dfdist


	def build_distdb(self,n_top=DEFAULT_N_TOP,n_vecs=MAX_NUM_VECS_TO_STORE,n_tops=[3,5,10,25,None]):
		dfdist=self.build_distmatrix(n_top=n_top,n_vecs=n_vecs)
		vocab=list(dfdist.index)

		for n_top in n_tops:
			self.log('>> gen distnet for n_top='+str(n_top))

			# insert?
			now=time.time()

			## NETDB
			g=Graph()
			# add props
			#vprop_d = g.new_vertex_property('object')
			#eprop_d = g.new_edge_property('object')
			vprop_id = g.vertex_properties['id'] = g.new_vertex_property('string')
			eprop_weight = g.edge_properties['weight'] = g.new_edge_property('float')
			eprop_cdist = g.edge_properties['cdist'] = g.new_edge_property('float')
			eprop_sim_rank = g.edge_properties['sim_rank'] = g.new_edge_property('float')
			# add nodes
			word2node={}
			for w in vocab:
				node = g.add_vertex()
				vprop_id[node] = w
				word2node[w]=node
			# add edges
			for word in tqdm(vocab):
				row = dfdist.loc[word].sort_values().iloc[1:n_top+1]
				sim_rank=0
				for word2,result in zip(row.index,row):
					sim_rank+=1
					edge = g.add_edge(word2node[word],word2node[word2])
					#eprop_d[edge] = {'weight':1-result,'sim_rank':sim_rank}
					eprop_cdist[edge]=result
					eprop_weight[edge] = 1-result
					eprop_sim_rank[edge] = sim_rank

			# save net
			dbfn=os.path.join(DB_DIR,self.id,DB_NAMESPACE_DISTS) + (f'.n_top={n_top}.gt' if db_fn else '.gt')
			g.save(dbfn,fmt='gt')
			print('>> saved: '+dbfn)

			tdist=round(time.time()-now,1)
			self.log('done building and inserting results in %ss' % tdist)



	## DATA ACCESS


	### GETTING VECTORS

	def get_vector_from_db(self,vecname):
		# print('Q:',vecname)
		#res=self.db.find_one({'word':vecname})
		#res=self.db.find_one({'word':vecname})
		
		# Tiny DB
		# Q=Query()
		# res=self.db.search(Q.word==vecname)
		# if not res: return None
		# res=res[0]
		# return np.array(res['vecs'])

		# just use gensim
		if vecname in self.gensim.vocab: return self.gensim[vecname]
		return None

	def get_all_word_vecs(self):
		all_words=[]
		all_vecs=[]
		
		# mongo
		#for d in tqdm(self.db.find().batch_size(1000),total=self.db.count()):
		#for d in self.db.find().batch_size(100):
		
		# tinydb
		# for d in self.db.all():
		# 	try:
		# 		all_words.append(d['word'])
		# 		all_vecs.append(d['vecs'])
		# 	except:
		# 		pass

		# just use gensim
		all_words=list(self.gensim.vocab)
		all_vecs=self.gensim.vectors

		return(all_words,all_vecs)


	def get_vector(self, word_or_formula, opts={}):
		# print('get_vector_opts',opts)
		formula_str=word_or_formula
		formula_str_q=formula_str.strip().replace('[','').replace(']','')
		self.log('looking for: '+formula_str_q)
		
		cached_formula_vec=self.get_vector_from_db(formula_str_q)
		if cached_formula_vec is not None: 
			self.log(f'\tfound cache of {formula_str_q}, returning')
			return cached_formula_vec

		self.log('\tdid not find cache, must be either a formula or alternatively dated')
		words_involved = split_words_only(word_or_formula)
		word2vecs = {}
		self.log('\tsplitting into components:' + ', '.join(words_involved))
		uncached_vecs=[]
	
		for w in words_involved:
			self.log(f'\t\tlooking for cache of word component: {w}')
			cached_word_vec = self.get_vector_from_db(w)
			if cached_word_vec is not None:
				self.log(f'\t\t\tfound cache of {w}, adding to component word2vecs dictionary')
				word2vecs[w]=cached_word_vec
			else:
				self.log(f'\t\t\tdid not find cache of "{w}", must be alt-dated')

				#  still no cache?
				#  maybe it has no period and we need to periodize
				#  try an average?
				
				word_vecs_to_avg = []
				w_periodized = periodize([w],self.periods)
				self.log('\t\t\t\treperiodized string into {0}'.format(', '.join(w_periodized)))
				for word_period in w_periodized:
					self.log(f'\t\t\t\tlooking for cache of word_period component {word_period}')
					cached_word_period_vec = self.get_vector_from_db(word_period)
					if cached_word_period_vec is not None:
						self.log(f'\t\t\t\t\tfound cache of "{word_period}", adding to word_vecs_to_avg')
						word_vecs_to_avg.append(cached_word_period_vec)
				# print('w?',word_vecs_to_avg)
				self.log('\t\t\t\taveraging {0} vectors for {1}'.format(len(word_vecs_to_avg), w))
				word_vec_avg = np.mean(word_vecs_to_avg,0) # column wise
				word2vecs[w]=word_vec_avg
		
		self.log('\treturning vector result for original formula: '+formula_str_q)
		vec_res = solve_vectors(formula_str_q,word2vecs)
		return vec_res

	def get_vectors(self,words_or_formulae):
		name2vecs = {}
		for word_or_formula in words_or_formulae:
			name2vecs[word_or_formula] = self.get_vector(word_or_formula)
			self.log('got vector for:'+word_or_formula)
		return name2vecs

	

	## DIST NET QUICK CALCS
	def expand_words_with_paths(self,words,sep='>',n_top=DEFAULT_N_TOP):
		print('expanding words from:',words)
		new_words=[]
		path_edges=[]
		gq=None
		for word_or_formula in words:
			if not sep in word_or_formula:
				new_words+=[word_or_formula]
			else:
				self.log('expanding words for formula: '+word_or_formula)
				w1,w2 = word_or_formula.split(sep,1)
				w1,w2=w1.strip(),w2.strip()
				self.log(f'finding shortest path from {w1} to {w2}')

				if not gq:
					if n_top:
						gq=self.distnet.filter_rank(n_top)
					else:
						gq=self.distnet
				
				path=self.distnet.shortest_path(w1,w2,g=gq)
				if path:
					new_words+=path
					for i,b in enumerate(new_words[1:]):
						a=new_words[i]
						path_edges+=[(a,b)]
				else:
					new_words+=[w1,w2]

		print('expanded words to:',new_words)
		return (new_words,path_edges)





	## GETTING SIMILARITY CALCULATIONS

	def get_most_similar(self,words,n_top=DEFAULT_N_TOP,periods=None,combine_periods=DEFAULT_COMBINED_PERIODS):
		if periods is None: periods=self.periods
		if combine_periods in {'simultaneous','diachronic'}:
			words=periodize(words, periods)

		(words,path_edges)=self.expand_words_with_paths(words,n_top=n_top)

		most_similar_data = []
		words_with_cached_dists = []
		words_with_uncached_dists = []

		for w in words:
			simdat = self.get_most_similar_from_distdb(w,n_top=n_top,combine_periods=combine_periods)
			if simdat:
				words_with_cached_dists.append(w)
				most_similar_data.extend(simdat)
			else:
				words_with_uncached_dists.append(w)

		self.log('found cached distance results for: '+', '.join(words_with_cached_dists))
		self.log('did not find cached distance results for: '+', '.join(words_with_uncached_dists))
		
		self.log('remaining input split into: ' + ', '.join(words_with_cached_dists))
		msd2=self.get_most_similar_by_vector(words_with_uncached_dists,n_top=n_top,combine_periods=combine_periods)
		most_similar_data.extend(msd2)

		## annotate for paths
		path_edge_set = set(path_edges)
		for d in most_similar_data:
			w1,w2=d['id'],d['id2']
			d['was_path'] = ((w1,w2) in path_edge_set)

		return most_similar_data





	def get_most_similar_from_distdb(self,word,n_top=DEFAULT_N_TOP,combine_periods=DEFAULT_COMBINED_PERIODS):
		# return [  (d['target'],d['weight']) for d in self.distdb.find({'source':word}) ]
		new_name_sims=[]


		print('get_most_similar_from_distdb('+word+')')

		for edge_d in self.distnet.most_similar_data(word,n_top=n_top):
			#if edge_d['sim_rank']>n_top: continue
			# print(edge_d)

			new_sim_d={}
			new_sim_d['id']=id1=word
			new_sim_d['id2']=id2=edge_d['word']
			worddat1=deperiodize_str(id1)
			worddat2=deperiodize_str(id2)
			new_sim_d['word'] = wordname1 = worddat1[0]
			new_sim_d['word2'] =wordname2 = worddat2[0]
			new_sim_d['period'] = period1 = worddat1[1]
			new_sim_d['period2'] = period2 = worddat2[1]
			new_sim_d['csim']=1-edge_d['cdist']
			new_sim_d['csim_rank']=int(edge_d['sim_rank'])
			# combine across periods?
			new_name_sims.append(new_sim_d)

		##  name_sims average?
		if combine_periods=='average' and new_name_sims:
			self.log('averaging results by period (`combine_periods` set to "average")')
			new_name_sims=average_periods(new_name_sims,val_key='csim',word_key='word2',period_key='period2')
		
		#print('final ld:')
		final_ld=new_name_sims[:n_top]
		#print(pd.DataFrame(final_ld))
		return final_ld


	def get_most_similar_by_vector(self,words,n_top=DEFAULT_N_TOP,combine_periods=DEFAULT_COMBINED_PERIODS):
		name2vec = self.get_vectors(words)
		self.log('got name2vec with %s vectors' % len(name2vec))
		return self.get_most_similar_by_vector_by_gensim(name2vec,n_top=n_top,combine_periods=combine_periods)

	
	def get_most_similar_by_vector_by_gensim(self,name2vec,n_top=DEFAULT_N_TOP,combine_periods=DEFAULT_COMBINED_PERIODS):
		all_sims=[]
		for name,vec in sorted(name2vec.items()):
			# print('name:',name)
			# print('vec:',vec)
			try:
				name_sims = self.gensim.wv.similar_by_vector(vec,topn=DEFAULT_N_TOP)  #(n_top*5)+1)
			except TypeError:
				continue
			self.log('got back from gensim for similar_by_vector: '+str(name_sims))

			new_name_sims=[]
			for xi,x in enumerate(name_sims[1:]):
				#print('x',x)
				match=x[0]
				csim=x[1]
				new_sim_d={}
				new_sim_d['id']=id1=name
				new_sim_d['id2']=id2=match

				# print('!?!?',id1,id2)

				worddat1=deperiodize_str(id1)
				worddat2=deperiodize_str(id2)
				new_sim_d['word'] = wordname1 = worddat1[0]
				new_sim_d['word2'] =wordname2 = worddat2[0]
				new_sim_d['period'] = period1 = worddat1[1]
				new_sim_d['period2'] = period2 = worddat2[1]
				new_sim_d['csim']=csim
				new_sim_d['csim_rank']=xi+1

				# combine across periods?
				new_name_sims.append(new_sim_d)

			#  name_sims average?
			if combine_periods=='average' and new_name_sims:
				new_name_sims=average_periods(new_name_sims,val_key='csim',word_key='word2',period_key='period2')

			all_sims.extend(new_name_sims[:n_top])
		
		self.log('collecting '+str(len(all_sims))+' cosine sims')
		return all_sims
	


# 	self.get_expanded_wordset = function(opts) {
# 		print('get_expanded_wordset()',opts)

# 		var expand_n=opts['expand_n']
# 		if(expand_n==undefined) { expand_n = DEFAULT_EXPAND_N }
# 		name2vecs = self.get_vectors(opts)
# 		log('retrieved vector data for existing words')
	  
# 		vecs = dict_values(name2vecs)
# 		sumvec = math.add(...vecs)
# 		words_already=opts['words']
# 		log('computed vector sum of existing words')

# 		opts['name2vec'] = {'sumvec':sumvec}
# 		most_similar_data = self.get_most_similar_by_vector(opts)
# 		log('found '+most_similar_data.length+' nearest words to sum vector')

# 		var matches = []
# 		most_similar_data.forEach(function(d) {
# 		   #  wordx=d.word2
# 		  #  don't include period anymore: should be an option?
# 		  wordx=d.word2.split('_')[0]
# 		  if(!words_already.includes(wordx)) {
			
			
# 			words_already.push(wordx)
# 			if(matches.length < expand_n) {
# 			  matches.push(wordx)
# 			}
# 		  }
# 		})
# 		return matches
# 	}

# 	#  self.progress(1.0,opts)
# 	return Model
# }





def split_words_only(_words):
	#return _words.replace('[','').replace(']','').replace(' ','').replace(',',' ').split()
	return [w for w in split_words_keep_punct(_words) if w and w[0].isalpha()]
	

def split_words_keep_punct(_words):
	import re
	return re.findall(r"[\w']+|[.,!?;\-\+\/\*\>]", _words)


def solve_vectors(formula, var2val={}):
	formula=formula.replace('[','').replace(']','')

	import expression
	parser = expression.Expression_Parser(variables=var2val)
	return parser.parse(formula)




# #  Umap
# function get_umap_from_vector_data(name2vec) {
# 	print(name2vec)

# 	data = []
# 	names = []
# 	for(name in name2vec) {
# 		data.push(name2vec[name])
# 		names.push(name)
# 	}
# 	print('names',names)

# 	umapjs = require('umap-js')

# 	const umap = new umapjs.UMAP({nComponents: 2,nEpochs: 400,nNeighbors: 3,});
# 	const embedding = umap.fit(data)

# 	out_ld = []

# 	embedding.forEach(function(erow,i) {
# 		name=names[i]
# 		word_period=deperiodize_str(name)
# 		word=word_period[0]
# 		period=word_period[1]

# 		out_d={}
# 		out_d['name']=names[i]
# 		out_d['word']=word
# 		out_d['period']=period
# 		out_d['umap_V1']=erow[0]
# 		out_d['umap_V2']=erow[1]
# 		out_ld.push(out_d)

# 		print(out_d)
# 	})


# 	return out_ld
# }




def periodize(words,periods):
	word_periods = []
	# print('periodize',words,periods)

	for w in words:
		path_piece = '>' in w
		if '_' in w and w.split('_')[1][0].isdigit(): 
			word_periods.append(w)
			continue  # if already period keep going
		
		word_pieces = split_words_keep_punct(w)
		for pi,p in enumerate(periods):
			if path_piece and pi>0: break
			# print(p,'...')
			if len(word_pieces)==1:
				word_period=w+'_'+p
			else:
				word_period_l = []
				for wpiece in word_pieces:
					if wpiece.isalpha():
							word_period_l.append(wpiece+'_'+p)
					else:
						word_period_l.append(wpiece)
				word_period=''.join(word_period_l)

			word_periods.append(word_period)
		# else:
			# word_periods.append(w)
	#print('PERIODIZED:',words,'-->',word_periods)
	return word_periods


def get_period_from(wordstr):
	if '_' in wordstr:
		period=wordsstr.split('_')[-1]
		if period[0].isdigit():
			return period
	return ''


def deperiodize_str(wordstr):
	new_word_pieces=[]
	word_pieces = split_words_keep_punct(wordstr)
	periods=[]
	for wpiece in word_pieces:
		if '_' not in wpiece:
			new_word_pieces.append(wpiece)
		else:
			word=wpiece.split('_')[0]
			period=wpiece.split('_')[1]
			periods.append(period)
			new_word_pieces.append(word)
	return (''.join(new_word_pieces), periods[0] if periods else '')


def average_periods(word_ld,val_key='csim',word_key='word',period_key='period',periods=None):
	import pandas as pd
	word_df=pd.DataFrame(word_ld)
	print('ORIG:\n',word_df.shape,'\n',word_df)
	if periods: word_df=word_df[word_df[period_key].isin(periods)]
	word_df_mean = word_df.groupby(word_key).mean().reset_index()
	word_df2=word_df.groupby(word_key).first().reset_index()
	lost_cols = list(set(word_df2.columns) - set(word_df_mean.columns))
	word_df_mean = word_df_mean.merge(word_df2[lost_cols+[word_key]], on=word_key,how='left')
	word_df_mean=word_df_mean[word_df.columns]

	print('AVGS:\n',word_df_mean.shape,'\n',word_df_mean.sort_values('csim_rank'))

	return word_df_mean.sort_values('csim_rank').to_dict('records')




# function average_periods(word_ld,val_key='csim',word_key='word',period_key='period',periods=undefined) {
# 	# create word2ld
# 	word2vals = {}
# 	word2eg = {}
# 	word_ld.forEach(function(word_d) {
# 		word=word_d[word_key]

# 		var ok = true
# 		if(periods!=undefined) {
# 			period=word_d[period_key]
# 			if(!periods.includes(period)) {
# 				ok=false
# 			}
# 		}

# 		if(ok) {
# 			if(!(word in word2vals)) { word2vals[word]=[]; word2eg[word]=word_d }
# 			word2vals[word].push(parseFloat(word_d[val_key]))
# 		}
# 	})

# 	word_old=[]
# 	for(word in word2vals) {
# 		word_vals = word2vals[word]
# 		word_vals_avg = math.mean(word_vals)
# 		word_od = {}
# 		for(k in word2eg[word]) { word_od[k]=word2eg[word][k] }
# 		word_od[val_key]=word_vals_avg
# 		word_old.push(word_od)
# 	}

# 	return word_old
# }


# async function get_orig_vocab(fn) {

# 	vocab_promise=new Promise(function(resolve,reject) { 

# 		var line_num=0
# 		var line_words=[]
# 		lineReader.eachLine(fn, function(line, last) {
# 			#  print('>>>',line_num,line.slice(0,5),last)
# 			if((line_num > 0) & (line!='')) {
# 				line_word=line.split(' ')[0]
# 				line_words.push(line_word)
# 				#  print(line_word)
# 			}
# 			line_num++

# 			if(last) {
# 				resolve(line_words)
# 			}
# 		})
		
# 	})
# 	vocab_result = await vocab_promise
# 	#  print('vocab_result!',vocab_result)
# 	return vocab_result
# }




# exports.with_model = with_model
# exports.W2V_MODELS = W2V_MODELS
# exports.periodize = periodize
# exports.deperiodize_str = deperiodize_str
# exports.get_umap_from_vector_data=get_umap_from_vector_data











if __name__=='__main__':
	pass

	e = Embedding(DEFAULT_W2V_MODEL,DEFAULT_W2V_FN,DEFAULT_PERIODS)
	#e.build_vecdb()
	#e.build_distmatrix()
	e.build_distdb()




	
	# print(e.get_vector('value_1800'))
	# print(e.get_vector('value'))
	# print(e.get_vector('value-importance'))
	# print(e.get_vector('value_1800-importance_1950'))


	# print(e.get_vectors(['value_1800','importance_1950','value']))


	# print(e.get_most_similar(['virtue','vice']))
	# print(e.get_most_similar(['virtue','vice']))
	# print(e.get_most_similar(['virtue','vice']))
	# print(e.get_most_similar(['virtue','vice']))
	# print(e.get_most_similar(['virtue','vice']))


	