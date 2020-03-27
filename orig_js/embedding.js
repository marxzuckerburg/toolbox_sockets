// Constants
DEFAULT_CORPUS = 'COHA'
DEFAULT_PERIOD_TYPE = 'byhalfcentury'
DEFAULT_N_STORE=100

W2V_MODELS = {
	'COHA_byhalfcentury_nonf': {
		"fn": "static/data/db/models/COHA_byhalfcentury_nonf/chained_full_combined/1800-2000.min=100.run=01.txt",
		'periods':['1800','1850','1900','1950'],
		'periods_nice':['1800-1850','1850-1900','1900-1950','1950-2000'],
		"corpus_desc":"COHA (Corpus of Historical American English), Non-Fiction"
	},

	'COHA_byhalfcentury_nonf_smpl': {
		"fn": "static/data/db/models/COHA_byhalfcentury_nonf/chained_combined/1800-1999.min=500.run=01.txt",
		'periods':['1800','1850','1900','1950'],
		'periods_nice':['1800-1850','1850-1900','1900-1950','1950-2000'],
		"corpus_desc":"COHA (Corpus of Historical American English), Non-Fiction [Samples]"
	},

	'COHA_bythirtyyear_nonf': {
		"fn": "static/data/db/models/COHA_bythirtyyear_nonf_full/chained_combined/1810-2020.min=100.run=01.txt",
		'periods':['1810','1840','1870','1900','1930','1960','1990'],
		'periods_nice':['1810-1840','1840-1870','1870-1900','1900-1930','1930-1960','1960-1990','1990-2020'],
		"corpus_desc":"COHA (Corpus of Historical American English), Non-Fiction"
	}
}

DEFAULT_W2V_MODEL = 'COHA_bythirtyyear_nonf'
DEFAULT_W2V_FN = W2V_MODELS[DEFAULT_W2V_MODEL]['fn']
DEFAULT_PERIODS = W2V_MODELS[DEFAULT_W2V_MODEL]['periods']
DEFAULT_EXPAND_N = 2
const DEFAULT_N_SIMILAR = 1


// word2vec imports
var w2v=require('word2vec')
const math = require('mathjs')
const path = require('path'); 
// var DataFrame = require('dataframe-js').DataFrame;

// load fields and variables
const fs = require('fs')
// const readline = require('readline')
var lineReader = require('line-reader')
let field2words = JSON.parse(fs.readFileSync('static/data/db/fields/_id2words.json')).data
let vec_names = JSON.parse(fs.readFileSync('static/data/db/fields/_vecids.json')).data
let content_words = JSON.parse(fs.readFileSync('static/data/db/misc/content_words.json')).data

// db import
const DB=require('./db.js')
const get_csim = require( 'compute-cosine-similarity' );

// global variables for storage
var fn2M = {}
var fn2voc={} 

// ------------ //

// fn=undefined,periods=undefined,voc_fn=undefined,log=console.log) {

async function with_model(opts,log=console.log,progress=console.log) {

	function Model() { }

	//fn_periods = opts2model_fn(opts)

	Model.fn = opts['model_fn']
	Model.periods=opts['model_periods']
	Model.model_id=opts['model_id']
	Model.opts=opts
	fn=Model.fn
	periods=Model.periods
	console.log('periods!',Model.periods)
	if(opts['voc_fn']==undefined) { opts['voc_fn']=fn.replace('.txt','.vocab.txt') }
	
	Model.voc_fn=opts['voc_fn']
	Model.log=log
	Model.progress=progress
	log('loading model: '+fn)

	model_vocab = await get_model(fn)
	
	// Model.progress(0.5,opts)

	Model.M = model_vocab[0]
	Model.vocab = model_vocab[1]
	Model.orig_vocab = await get_orig_vocab(fn)
	// Model.db = DB.get_vecdb(opts['model_id'])


	Model.num_words = function() {
		return this.vocab.length
	}

	Model.get_db = function() { return DB.get_vecdb(this.model_id) }
	Model.get_distdb = function() { return DB.get_distdb(this.model_id) }


	Model.build_vecdb = async function(opts) {
		db=Model.get_db()
		// distdb=Model.get_distdb()
		all_keys=[]
		this.orig_vocab.forEach(async function(word,i) {
			console.log(i,word)
			db_key=word
			all_keys.push(db_key)
			db_val=this.M.getVector(word).values
			// db_val2=this.M.mostSimilar(word, DEFAULT_N_STORE)

			// console.log('db_key',db_key)
			// console.log('db_val',db_val)
			await db.set(db_key,db_val)
			// await distdb.set(db_key,JSON.stringify(db_val2))
			// console.log('')
		})
		await db.set('_keys',all_keys)
		// db.get('virtue_1800').then(function(x) { console.log('RESULT:',x) })
		// throw 1
	}

	Model.get_vector = async function(word_or_formula, opts={}) {
		// console.log('get_vector_opts',opts['word'])
		db=Model.get_db()
		// word_or_formula = opts['word']
		console.log('get_vector_opts',opts)
		formula_str=word_or_formula
		Model.log('locating vector position for: '+formula_str)
		formula_str_q=formula_str.trim().split('[').join('').split(']').join('')
		

		cached_formula_vec=await db.get(formula_str_q)
		// console.log('cached_formula_vec:::',opts['word'],formula_str_q,cached_formula_vec)

		if(cached_formula_vec!=undefined) {  return cached_formula_vec }


		var words_involved = split_words_only(word_or_formula)
		var word2vecs = {}
		var uncached_vecs=[]
		// console.log('words_involved',words_involved)

		for(wi=0; wi<words_involved.length; wi++) {
			w=words_involved[wi]
			cached_word_vec = await db.get(w)
			//console.log('cached_word_vec!',w,cached_word_vec)
			if(cached_word_vec!=undefined) {
				word2vecs[w]=cached_word_vec
				// console.log('word2vecs!?!?!',w,word2vecs)
			} else {
				// still no cache?
				// maybe it has no period and we need to periodize
				// try an average?
				
				word_vecs_to_avg = []
				//console.log('w???',w,words_involved)
				w_periodized = periodize([w],opts['periods'])
				//console.log('w_periodized',w_periodized)
				
				for(wpi=0; wpi<w_periodized.length; wpi++) {
					word_period=w_periodized[wpi]
					console.log(word_period,'!?')
					// try cache one last time
					cached_word_period_vec = await db.get(word_period)
					//console.log('cached_word_period_vec',word_period,cached_word_period_vec,'!?')
					if(cached_word_period_vec!=undefined) {
						word_vecs_to_avg.push(cached_word_period_vec)
					}
				}

				console.log('w?',word_vecs_to_avg)
				word_vec_avg = math.add(...word_vecs_to_avg)
				word2vecs[w]=word_vec_avg
			}
		}

		// calc formula?
		// console.log('word2vecs',word2vecs)
		vec_res = solve_vectors(formula_str_q,word2vecs)
		// console.log('vec_res!',formula_str_q,vec_res)
		// vec_res = await formula_promise
		return vec_res
	}



	Model.get_vectors = async function(opts) {
		var name2vecs = {}
		words=opts['words']
		for(wii=0; wii<words.length; wii++) {
			word_or_formula=words[wii]
			// opts['word']=word_or_formula
			console.log('WORDDDD',word_or_formula,wii,words)
			// try {
			name2vecs[word_or_formula] = await Model.get_vector(word_or_formula, opts)
			console.log('got_vector',word_or_formula,name2vecs[word_or_formula])
			// } catch(err) {
				// console.log('err getting vector for!',word_or_formula,err)
			// }
		}


		// opts['words'].forEach(async function(word_or_formula,i) {
		// 	// Model.progress(i/opts['words'].length,opts)
		// 	opts['word']=word_or_formula
		// 	try {
		// 		name2vecs[word_or_formula] = await Model.get_vector(opts)

		// 		console.log('got_vector',word_or_formula,name2vecs[word_or_formula])
		// 	} catch(err) {
		// 		console.log('err getting vector for!',word_or_formula,err)
		// 	}
		// });
		console.log('get_vectors returning name2vecs:',name2vecs)
		return name2vecs
	}

	Model.get_most_similar = async function(opts) {
		console.log('most_similar_opts:', opts)

		if(opts['combine_periods']=='simultaneous' | opts['combine_periods']=='diachronic') {
			opts['words']=periodize(opts['words'], opts['periods'])
		}

		Model.log('input split into: ' + opts['words'].join(', '))
		name2vec = await Model.get_vectors(opts)
		// console.log('FINAL NAME2VEC',name2vec)
		
		most_similar_data = await Model.get_most_similar_by_vector(name2vec,opts)
		return most_similar_data
	}


	Model.get_most_similar2 = async function(opts) {
		console.log('most_similar_opts:', opts)
		n_top=opts['n_top']
		if(n_top==undefined) { n_top = DEFAULT_N_SIMILAR }

		// periodize?
		if(opts['combine_periods']=='simultaneous' | opts['combine_periods']=='diachronic') {
			opts['words']=periodize(opts['words'], opts['periods'])
		}
		Model.log('input split into: ' + opts['words'].join(', '))

		// first check dist db
		distdb=Model.get_distdb()
		most_similar_data = []
		words_with_cached_dists = []
		words_with_uncached_dists = []
		words=opts['words']

		// already have cache for?

		for(_i=0; _i<words.length; _i++) {
			w=words[_i]
			console.log(_i,w)
			distcache = await distdb.get(w)
			if(distcache) {
				words_with_cached_dists.push(w)
				distcache=JSON.parse(distcache)
				simdat=distcache.slice(0,n_top)
				most_similar_data.push(simdat)
			} else {
				words_with_uncached_dists.push(w)
			}
		}

		Model.log('found cached distance results for: '+words_with_cached_dists.join(', '))
		Model.log('did not find cached distance results for: '+words_with_uncached_dists.join(', '))
		
		opts['words']=words_with_uncached_dists
		name2vec = await Model.get_vectors(opts)
		// console.log('FINAL NAME2VEC',name2vec)
		
		// add uncached results
		most_similar_data.push(...await Model.get_most_similar_by_vector(name2vec,opts))

		return most_similar_data
	}




	Model.get_current_vec_names = async function (opts={}) {
		return await this.get_db().get('_keys')
	}


	Model.get_nearest_words_with_db = async function(name2vec, n_top=25, opts={}, n_store=1000) {
		vecdb = Model.get_vecdb()

	}



	Model.get_nearest_words_with_db0 = async function(name2vec, n_top=25, opts={}, n_store=1000) {
		db=Model.get_db()
		
		vecnames = await Model.get_current_vec_names(opts)
		console.log('vecnames!',vecnames)

		// closest_so_far = []

		var closest_so_far={}
		for (var vn in name2vec) { closest_so_far[vn]=[] }

		for(vci=0; vci<vecnames.length; vci++) {
			if(vci%1000 == 0) { console.log(vci,vecnames[vci],'...') }
			for(var vecname1 in name2vec) {
				vec1=name2vec[vecname1]

				vecname2=vecnames[vci]
				// console.log(vci,vecname2,vecname1,'...')
				// vec2=await Model.get_vector(vecname2,opts=opts)
				
				vec2=await db.get(vecname2)

				try {
					dist=1-get_csim(vec1,vec2)
					closest_so_far[vecname1].push([vecname2,dist])
				} catch(err) {
					console.log('ERROR::',err)					
					console.log('vec1',vecname1,vec1)
					console.log('vec2',vecname2,vec2)
					console.log('dist',dist,'\n')
				}
				

				// vecname2dists

				// if(closest_so_far.length > n_top) {
				// if(closest_so_far[vecname1].length > n_store) {
				// 	closest_so_far[vecname1].sort(function(a, b) { return a[1] - b[1]; })	
				// 	closest_so_far[vecname1] = closest_so_far[vecname1].slice(0,n_top)
				// }
			}
		}

		sims=[]
		// distdb=Model.get_distdb()
		for(var vecname in closest_so_far) {
			// console.log('>>VEC',vecname,'...')
			vecdists=closest_so_far[vecname]
			vecdists.sort(function(a, b) { return a[1] - b[1] })
			vecdists=vecdists.slice(0,n_top)


			// distdb.set(vecname, JSON.stringify(vecdists))
			for(vii=0; vii<n_top; vii++) {
				vecname2=vecdists[vii][0]
				dist=vecdists[vii][1]
				new_sim_d={}
				new_sim_d['id']=id1=vecname
				new_sim_d['id2']=id2=vecname2
				worddat1=deperiodize_str(id1)
				worddat2=deperiodize_str(id2)
				new_sim_d['word'] = wordname1 = worddat1[0]
				new_sim_d['word2'] =wordname2 = worddat2[0]
				new_sim_d['period'] = period1 = worddat1[1]
				new_sim_d['period2'] = period2 = worddat2[1]
				new_sim_d['csim']=dist
				sims.push(new_sim_d)
			}
		}
		return sims
	}



	Model.get_most_similar_by_vector = async function(name2vec,opts={}) {
		console.log('OPTS!','get_most_similar_by_vector',opts)
		var name2vec=name2vec
		var n_top=opts['n_top']
		var periods=opts['periods']
		if(n_top==undefined) { n_top = DEFAULT_N_SIMILAR }

		all_sims = []
		n_names = 0
		for(var name in name2vec) { n_names++ }
		i_names=0

		sims = await Model.get_nearest_words_with_db(name2vec, n_top=n_top, opts=opts)



		for(var name in name2vec) {
			// Model.progress(i_names/n_names, opts)
			i_names++
			name_sims=[]
			unique_words=new Set()
			Model.log('getting '+ n_top +' nearest word vectors to: ' + name)


			// first check distdb
			// csim_cache = distdb.get(name)
			// if(csim_cache!=undefined) {
			// 	return 
			// }


			vec=name2vec[name]
			//console.log('vec!',name,vec)
			// sims = this.M.getNearestWords(vec, (n_top+1)*5)
			sims = await Model.get_nearest_words_with_db(vec, opts=opts)
			console.log('SIMS:',sims)
			
			// sims = this.M.getNearestWords(vec, (n_top+1)*2)
			
			sims.forEach(function(sim_d) {
				new_sim_d={}
				new_sim_d['id']=id1=name
				new_sim_d['id2']=id2=sim_d['word']
				worddat1=deperiodize_str(id1)
				worddat2=deperiodize_str(id2)
				new_sim_d['word'] = wordname1 = worddat1[0]
				new_sim_d['word2'] =wordname2 = worddat2[0]
				new_sim_d['period'] = period1 = worddat1[1]
				new_sim_d['period2'] = period2 = worddat2[1]
				new_sim_d['csim']=sim_d['dist']
				if((period1!=undefined) & (opts['combine_periods']=='diachronic') & (period2!=period1)) {
					// skip because let's not compare across periods in that case
					// console.log('yep!!')
				} else {

					// console.log('new_sim_d!?',new_sim_d)
					if((!(id2 in name2vec)) &(!(wordname2 in name2vec)) &(unique_words.size<n_top)) {
						// if we either want all periods, or periods wanted includes this one
						if(periods==undefined | (periods.includes(period2))) {
							// start a new dictionary
							name_sims.push(new_sim_d)


							// console.log('new_sim_d',new_sim_d)
							if(!(unique_words.has(wordname2))) {
								unique_words.add(wordname2)
							}
						
						}
					}
				}
			})

			// final average
			if(opts['combine_periods']=='average') { name_sims=average_periods(name_sims,val_key='csim',word_key='word2',period_key='period2') }

			all_sims.push(...name_sims)
		}
		// console.log('all_sims',all_sims.length)
		return all_sims
	}

	// Model.get_most_similar_by_vector = function(opts) {
	// 	console.log('OPTS!','get_most_similar_by_vector',opts)
	// 	var name2vec=opts['name2vec']
	// 	var n_top=opts['n_top']
	// 	var periods = opts['periods']
	// 	if(n_top==undefined) { n_top = DEFAULT_N_SIMILAR }


	// 	all_sims = []
	// 	n_names = 0
	// 	for(var name in name2vec) { n_names++ }
	// 	i_names=0


	// 	for(var name in name2vec) {
	// 		// Model.progress(i_names/n_names, opts)
	// 		i_names++
	// 		Model.log('getting '+ n_top +' nearest word vectors to: ' + name)
	// 		vec=name2vec[name]
	// 		//console.log('vec!',name,vec)
	// 		sims = this.M.getNearestWords(vec, (n_top+1)*5)
	// 		// sims = this.M.getNearestWords(vec, (n_top+1)*2)
	// 		name_sims=[]
	// 		unique_words=new Set()
	// 		sims.forEach(function(sim_d) {
	// 			new_sim_d={}
	// 			new_sim_d['id']=id1=name
	// 			new_sim_d['id2']=id2=sim_d['word']
	// 			worddat1=deperiodize_str(id1)
	// 			worddat2=deperiodize_str(id2)
	// 			new_sim_d['word'] = wordname1 = worddat1[0]
	// 			new_sim_d['word2'] =wordname2 = worddat2[0]
	// 			new_sim_d['period'] = period1 = worddat1[1]
	// 			new_sim_d['period2'] = period2 = worddat2[1]
	// 			new_sim_d['csim']=sim_d['dist']
	// 			if((period1!=undefined) & (opts['combine_periods']=='diachronic') & (period2!=period1)) {
	// 				// skip because let's not compare across periods in that case
	// 				// console.log('yep!!')
	// 			} else {

	// 				// console.log('new_sim_d!?',new_sim_d)
	// 				if((!(id2 in name2vec)) &(!(wordname2 in name2vec)) &(unique_words.size<n_top)) {
	// 					// if we either want all periods, or periods wanted includes this one
	// 					if(periods==undefined | (periods.includes(period2))) {
	// 						// start a new dictionary
	// 						name_sims.push(new_sim_d)


	// 						// console.log('new_sim_d',new_sim_d)
	// 						if(!(unique_words.has(wordname2))) {
	// 							unique_words.add(wordname2)
	// 						}
						
	// 					}
	// 				}
	// 			}
	// 		})

	// 		// final average
	// 		if(opts['combine_periods']=='average') { name_sims=average_periods(name_sims,val_key='csim',word_key='word2',period_key='period2') }

	// 		all_sims.push(...name_sims)
	// 	}
	// 	// console.log('all_sims',all_sims.length)
	// 	return all_sims
	// }

	Model.get_expanded_wordset = function(opts) {
		console.log('get_expanded_wordset()',opts)

		var expand_n=opts['expand_n']
		if(expand_n==undefined) { expand_n = DEFAULT_EXPAND_N }
		name2vecs = Model.get_vectors(opts)
		log('retrieved vector data for existing words')
	  
		vecs = dict_values(name2vecs)
		sumvec = math.add(...vecs)
		words_already=opts['words']
		log('computed vector sum of existing words')

		opts['name2vec'] = {'sumvec':sumvec}
		most_similar_data = Model.get_most_similar_by_vector(opts)
		log('found '+most_similar_data.length+' nearest words to sum vector')

		var matches = []
		most_similar_data.forEach(function(d) {
		   // wordx=d.word2
		  // don't include period anymore: should be an option?
		  wordx=d.word2.split('_')[0]
		  if(!words_already.includes(wordx)) {
			
			
			words_already.push(wordx)
			if(matches.length < expand_n) {
			  matches.push(wordx)
			}
		  }
		})
		return matches
	}

	// Model.progress(1.0,opts)
	return Model
}






// Get vocabulary from a vocab fn
function get_vocab(fn) {
	return new Promise(function(resolve,reject) {
		
		fs.readFile(fn, "UTF8", function(err, data) {
			var all_vocab =[]
			vtxt=data
			var all_vocab=[]
			// console.log(fn)
			lines=vtxt.split('\n')
			lines.forEach(function(line) {
				// console.log(line)
				word=line.split(' ')[0]
				//console.log(word)
				all_vocab.push(word)
			});
			// console.log('all_vocab',all_vocab)
			resolve(all_vocab)
		})
		
	});
}


// get model (as a promise)
async function get_model(w2v_fn = DEFAULT_W2V_FN) { 
	// console.log('>> loading w2v_fn:',w2v_fn)
	var voc_fn=w2v_fn.replace('.txt','.vocab.txt')
	var model_promise
	var vocab_promise
	if(w2v_fn in fn2M) {
		console.log('>> RESTORING FROM CACHE:',w2v_fn)
		vocab_promise = new Promise(function(res,rej) { res(fn2voc[w2v_fn]) })
		model_promise = new Promise(function(res,rej) { res(fn2M[w2v_fn]) })
	} else {
		model_promise = new Promise(function(res,rej) { 
			w2v.loadModel(w2v_fn,function(err,model) {
				fn2M[w2v_fn]=model
				//console.log('loaded',w2v_fn,model)
				res(model)
			})
		})
		vocab_promise = new Promise(function(res,rej) { 
			get_vocab(voc_fn).then(function(all_vocab) {
				fn2voc[w2v_fn]=all_vocab
				res(all_vocab)
			})
		})
	}
	M = await model_promise
	Voc = await vocab_promise
	//console.log("mvoc2",M,Voc)
	return [M,Voc]
}


// function opts2model_fn(opts) {
// 	console.log(opts,'??????')
// 	console.log('model_id: ',opts['model_id'])
// 	res=W2V_MODELS[opts['model_id']]
// 	console.log('model res:',res)
// 	return res
// }








function split_words_only(_words) {
	_words_l=_words.split(/[^A-Z_\[\]a-z0-9]+/);
	_words_l2=[]
	_words_l.forEach(function(w) { if(w) { _words_l2.push(w) } })
	return _words_l2
}

function split_words(_words) {
	console.log('split_words',_words)
	_words=_words.replace('\r\n',',').replace('\r',',').replace('\n',',')
	try {
		_words_l0 = _words.split(',')
	} catch(TypeError) {
		return [];
	}
	_words_l = []
	for(wii=0; wii<_words_l0.length; wii++) {
		_words_l.push(_words_l0[wii].trim());
	}
	return _words_l;
}

function split_words_keep_punct(_words) {
	return _words.match(/\\[^]|\.{3}|\w+|[^\w\s]/g)
}

function isAlpha(str) {
  return /^[a-zA-Z]+$/.test(str);
}


function reformat_formula_str(_words) {
	var _words=_words.replace(' ','')
	return _words
}

function compute_arrays(x, y, operator) {
	//console.log('computing array with operator',operator)
	//console.log(x,operator,y)


	if(operator=='+') { return math.add(x,y) }
	if(operator=='-') { return math.subtract(x,y) }
	if(operator=='*') { return math.multiply(x,y) }
	if(operator=='/') { return math.divide(x,y) }
}

function vector_add(x, y) { return math.add(x,y) }
function vector_subtract(x, y) { return math.subtract(x,y) }
function vector_divide(x, y) { return math.divide(x,y) }
function vector_multiply(x, y) { return math.multiply(x,y) }


function solve_vectors(formula, var2val={}) {
	formula=formula.replace('[','').replace(']','')
	const expr = require('expression-eval');
	var ast = expr.parse('('+formula+')');
	return compute_tree(ast,var2val=var2val);
}

function compute_tree(tree,var2val={}) {
	var arr_left=undefined;
	var arr_right=undefined;
	var op=undefined;
	var new_val=undefined;
	// if branches
	if (tree.left) { 
		// console.log('found split in tree:',tree.left,tree.operator,tree.right)
		arr_left = compute_tree(tree.left,var2val)
		op = tree.operator
		arr_right = compute_tree(tree.right,var2val)
		new_val = compute_arrays(arr_left,arr_right,op)
		// console.log('computation result ',arr_left[0],op,arr_right[0],'= ',new_val[0])
		return new_val;
	} else if (tree.value) {
		// console.log('found a value in tree',tree.value)
		return tree.value;
	} else {
		// console.log('found a variable in tree',tree.name)
		new_val = var2val[tree.name]
		// console.log('found a new_val of',new_val) //.islice(0,5))
		//console.log('new_val',new_val)
		return new_val;
	}
}

function dict_values(vector_data) {
	var vecs = []
	for(var name in vector_data) { 
		vec = vector_data[name]
		if(vec!=undefined) { 
			vecs.push(vec)
		}
	}
	return vecs
}



// Umap
function get_umap_from_vector_data(name2vec) {
	console.log(name2vec)

	data = []
	names = []
	for(name in name2vec) {
		data.push(name2vec[name])
		names.push(name)
	}
	console.log('names',names)

	umapjs = require('umap-js')

	const umap = new umapjs.UMAP({nComponents: 2,nEpochs: 400,nNeighbors: 3,});
	const embedding = umap.fit(data)

	out_ld = []

	embedding.forEach(function(erow,i) {
		name=names[i]
		word_period=deperiodize_str(name)
		word=word_period[0]
		period=word_period[1]

		out_d={}
		out_d['name']=names[i]
		out_d['word']=word
		out_d['period']=period
		out_d['umap_V1']=erow[0]
		out_d['umap_V2']=erow[1]
		out_ld.push(out_d)

		console.log(out_d)
	})


	return out_ld
}



// M = new Model(DEFAULT_W2V_FN)
// M = gen_model(DEFAULT_W2V_FN)
// console.log('Mfn',M.fn)

// gen_model(DEFAULT_W2V_FN).then(function(M) {
// 	console.log('M??',M)
// 	console.log('Mfn2',M.fn)
// 	console.log("Mvoclength",M.num_words())
	// console.log('Mvec11',M.get_vectors(['word_1950', 'word_1950 + word_1900', 'value_1800']))
	// console.log('Mvec22',M.get_most_similar('value_1800,value_1800-value_1950'))
	// console.log('MM',M.M)
// })

// console.log('MVOC',M.vocab)


function periodize(words,periods) {
	word_periods = []
	console.log('periodize',words,periods)
	words.forEach(function(w) {
		if(!w.includes('_')) {
			word_pieces = split_words_keep_punct(w)
			periods.forEach(function(p) {
				var word_period
				if(word_pieces.length==1) {
					word_period=w+'_'+p
				} else {
					word_period_l = []
					word_pieces.forEach(function(wpiece) {
						if(isAlpha(wpiece)) {
							word_period_l.push(wpiece+'_'+p)
						} else {
							word_period_l.push(wpiece)
						}
					})
					word_period=word_period_l.join('')
				}


				word_periods.push(word_period)
			})
		} else {
			word_periods.push(w)
		}
	})

	console.log('PERIODIZED:',words,'-->',word_periods)
	return word_periods
}

function get_period_from(wordstr) {
	if(wordstr.includes('_')) {
		return wordsstr.split('_').slice(-1)[0]
	}
	return ''
}

function deperiodize_str(wordstr) {
	new_word_pieces=[]
	word_pieces = split_words_keep_punct(wordstr)
	periods=[]
	word_pieces.forEach(function(wpiece) {
		if(!wpiece.includes('_')) {
			new_word_pieces.push(wpiece)
		} else {
			word=wpiece.split('_')[0]
			period=wpiece.split('_')[1]
			periods.push(period)
			new_word_pieces.push(word)
		}
	})
	return [new_word_pieces.join(''), periods[0]]
}



function average_periods(word_ld,val_key='csim',word_key='word',period_key='period',periods=undefined) {
	//create word2ld
	word2vals = {}
	word2eg = {}
	word_ld.forEach(function(word_d) {
		word=word_d[word_key]

		var ok = true
		if(periods!=undefined) {
			period=word_d[period_key]
			if(!periods.includes(period)) {
				ok=false
			}
		}

		if(ok) {
			if(!(word in word2vals)) { word2vals[word]=[]; word2eg[word]=word_d }
			word2vals[word].push(parseFloat(word_d[val_key]))
		}
	})

	word_old=[]
	for(word in word2vals) {
		word_vals = word2vals[word]
		word_vals_avg = math.mean(word_vals)
		word_od = {}
		for(k in word2eg[word]) { word_od[k]=word2eg[word][k] }
		word_od[val_key]=word_vals_avg
		word_old.push(word_od)
	}

	return word_old
}


async function get_orig_vocab(fn) {

	vocab_promise=new Promise(function(resolve,reject) { 

		var line_num=0
		var line_words=[]
		lineReader.eachLine(fn, function(line, last) {
			// console.log('>>>',line_num,line.slice(0,5),last)
			if((line_num > 0) & (line!='')) {
				line_word=line.split(' ')[0]
				line_words.push(line_word)
				// console.log(line_word)
			}
			line_num++

			if(last) {
				resolve(line_words)
			}
		})
		
	})
	vocab_result = await vocab_promise
	// console.log('vocab_result!',vocab_result)
	return vocab_result
}




exports.with_model = with_model
exports.W2V_MODELS = W2V_MODELS
exports.periodize = periodize
exports.deperiodize_str = deperiodize_str
exports.get_umap_from_vector_data=get_umap_from_vector_data







