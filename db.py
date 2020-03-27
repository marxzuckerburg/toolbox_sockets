M2VDB = {}
M2DDB = {}

from app_config import *

def get_db():
	from pymongo import MongoClient
	client = MongoClient()
	db = client[DB_NAME]
	return db


def get_vecdb(model_id):
	if not model_id in M2VDB:
		tbl_name=DB_NAMESPACE_VECS + '__' + model_id
		# print('>> connecting to: '+tbl_name)
		db=get_db()
		M2VDB[model_id] = tbl = db[tbl_name]
		tbl.create_index('word')
	return M2VDB[model_id]

def get_distdb(model_id):
	if not model_id in M2DDB:
		db=get_db()
		#M2DDB[model_id] = tbl = db[DB_NAMESPACE_DISTS]   # no model id since maybe we'll connect words across models
		M2DDB[model_id] = tbl = db[DB_NAMESPACE_DISTS+'__'+model_id]   # no model id since maybe we'll connect words across models
		tbl.create_index('source')
		tbl.create_index('target')
	return M2DDB[model_id]


