const cutoff=DEFAULT_CSIM_CUTOFF

function ld2dld(ld,key) {
	var dld={}
	ld.forEach(function(d) {
		k=d[key]
		if(!(k in dld)) { dld[k]=[] }
		dld[k].push(d)
	})
	return dld
}


function sims2net(most_similar_data,opts={}) {
	nets=[]
	console.log('sims2net opts',opts)
	console.log('MSDMSD',most_similar_data)
	if(opts['combine_periods']=='diachronic') {
		most_similar_data_by_period = ld2dld(most_similar_data,'period')
		for(period in most_similar_data_by_period) {
			period_data = most_similar_data_by_period[period]
			console.log('period!!',period,period_data)
			net = mostsim2netjson(period_data,opts=opts)
			netd = {'period':period, 'netdata':net}
			nets.push(netd)
		}
	} else {
		net = mostsim2netjson(most_similar_data,opts=opts)
		netd = {'period':undefined, 'netdata':net}
		nets.push(netd)
	}

	console.log('NETSSS:',nets)

	return nets
}


function mostsim2netjson(most_similar_data,opts={}) {
	// format data
	var nodes = []
	var nodes_sofar = []
	var links = []

	most_similar_data.forEach(function(data,i) {
		// console.log(i,most_similar_data.length)
		// progress(i/most_similar_data.length, opts)
		node1={'id':data['id'], 'word':data['word'], 'period':data['period']}
		node2={'id':data['id2'], 'word':data['word2'], 'period':data['period2']}
		word1=data['id']
		word2=data['id2']
		csim=data['csim']

		// console.log('ndoeword',word1,word2,node1,node2)

		if (cutoff==undefined | csim>=cutoff) {
			maybe_new_nodes = [[word1,node1], [word2,node2] ]
			maybe_new_nodes.forEach(function(dat) { 
				_word=dat[0]
				_node=dat[1]
			  	if(!(nodes_sofar.includes(_word))) {
					nodes_sofar.push(_word)
					nodes.push(_node)
				}
			});


		
			// console.log(word1,'--',csim,'-->',word2)
			links.push({'source':word1, 'target':word2, 'weight':csim})
		}
	});

	//console.log('nodes',nodes)
	//console.log('links',links)

	//return (nodes,links);
	return {'nodes':nodes, 'links':links}
}





exports.mostsim2netjson = mostsim2netjson
exports.sims2net = sims2net