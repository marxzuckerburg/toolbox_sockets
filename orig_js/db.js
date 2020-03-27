
NAMESPACE_VECS = 'vectors'


const Keyv = require('keyv')
// const levelgraph = require("levelgraph")

// const keyv = new Keyv(DB_CONN_STR)
// const vecdb = new Keyv(DB_CONN_STR, {})

var M2VDB = {}
var M2DDB = {}
function  get_vecdb_redis(model_id) {
	if(!(model_id in M2VDB)) { M2VDB[model_id] = new Keyv(DB_CONN_STR, {'namespace':'vecdb:'+model_id}) }
	return M2VDB[model_id]
}

function get_distdb(model_id) {
	// dbfn='db/'+model_id+'.ldb'
	if(!(model_id in M2VDB)) { 
		// M2VDB[model_id] = levelgraph(dbfn)
		M2VDB[model_id] = new Keyv(DB_CONN_STR, {'namespace':'distdb:'+model_id})
	}
	return M2VDB[model_id]
}

async function get_vecdb(model_id) {
	const MongoClient = require('mongodb').MongoClient
	const url = 'mongodb://localhost:27017'
	
	return await new Promise(function(resolve,reject) { 
		
		if(model_id in M2VDB) { resolve(M2VDB[model_id]) }

		MongoClient.connect(url, function(err, client) {
			
			dbtbl = client.db('vecdb').collection(model_id)
			console.log('dbtbl!',dbtbl)
			M2VDB[model_id]=dbtbl
			resolve(dbtbl)
		 
		  client.close()
		})
	})
}



async function test() {
	vecdb.set('model_word_1', [1,2,3,4,5,6,7,8,9,10])

	// console.log(await vecdb.get('model_word_1'))
}







exports.get_vecdb=get_vecdb
exports.get_distdb=get_distdb