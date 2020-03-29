from graph_tool.all import *

from app_config import *
import os

M2VDB = {}
M2DDB = {}



def get_distdb(model_id):
	return get_distdb_gt(model_id)





def get_distdb_gt(model_id):
	if not model_id in M2DDB:
		dbdir=os.path.join(DB_DIR,model_id)
		dbfn=os.path.join(DB_DIR,model_id,DB_NAMESPACE_DISTS+'.gt')
		if not os.path.exists(dbdir): os.makedirs(dbdir)

		if os.path.exists(dbfn):
			g = load_graph(dbfn,fmt='gt')
		else:
			g=Graph()

		M2DDB[model_id] = g

	return M2DDB[model_id]


def get_distdb_cog(model_id):
	if not model_id in M2DDB:
		from cog.torque import Graph
		dbdir=os.path.join(DB_DIR,model_id,DB_NAMESPACE_DISTS)
		dbfn=os.path.join(DB_DIR,model_id,DB_NAMESPACE_DISTS+'.edgelist.txt')
		if not os.path.exists(dbdir): os.makedirs(dbdir)

		g = Graph(graph_name=DB_NAMESPACE_DISTS, cog_dir=dbdir)

		if os.path.exists(dbfn): return g.load_edgelist(dbfn, DB_NAMESPACE_DISTS)
		
		M2DDB[model_id] = g
	return M2DDB[model_id]




def get_vecdb_tinydb(model_id):
	from tinydb import TinyDB
	if not model_id in M2VDB:
		dbdir=os.path.join(DB_DIR,model_id)
		dbfn=os.path.join(dbdir,DB_NAMESPACE_VECS+'.json')
		if not os.path.exists(dbdir): os.makedirs(dbdir)
		db = M2VDB[model_id] = TinyDB(dbfn)
	return M2VDB[model_id]


def get_distdb_tinydb(model_id):
	from tinydb import TinyDB
	if not model_id in M2DDB:
		dbdir=os.path.join(DB_DIR,model_id)
		dbfn=os.path.join(DB_DIR,model_id,DB_NAMESPACE_DISTS+'.json')
		if not os.path.exists(dbdir): os.makedirs(dbdir)
		db = M2DDB[model_id] = TinyDB(dbfn)
	return M2DDB[model_id]


def get_db_mongo():
	from pymongo import MongoClient
	client = MongoClient()
	db = client[DB_NAME]
	return db


def get_vecdb_mongo(model_id):
	global M2VDB

	if not model_id in M2VDB:
		tbl_name=DB_NAMESPACE_VECS + '__' + model_id
		# print('>> connecting to: '+tbl_name)
		db=get_db()
		M2VDB[model_id] = tbl = db[tbl_name]
		tbl.create_index('word')
	return M2VDB[model_id]

def get_distdb_mongo(model_id):
	global M2DDB

	if not model_id in M2DDB:
		db=get_db()
		#M2DDB[model_id] = tbl = db[DB_NAMESPACE_DISTS]   # no model id since maybe we'll connect words across models
		M2DDB[model_id] = tbl = db[DB_NAMESPACE_DISTS+'__'+model_id]   # no model id since maybe we'll connect words across models
		tbl.create_index('source')
		tbl.create_index('target')
	return M2DDB[model_id]
