### Constants
DEFAULT_WORD_STR='data,work,culture'


# GLOBAL_OPTS = {'fields':get_fields(), 'vecs':VECS, 'x_vec':'umap_V3', 'y_vec':'umap_V4'}
# GLOBAL_OPTS['all_fields_vecs'] = sorted(list(set(GLOBAL_OPTS['fields'] + GLOBAL_OPTS['vecs'])))
GLOBAL_OPTS={}
GLOBAL_OPTS['points']='movement'
GLOBAL_OPTS['view']='spaces'
GLOBAL_OPTS['x_vec_str']='King - Man + Woman'
GLOBAL_OPTS['y_vec_str']='Young - Old'


# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None
#async_mode = 'gevent'

DEFAULT_CORPUS = 'COHA'
DEFAULT_PERIOD_TYPE = 'byhalfcentury'
DEFAULT_N_STORE=100
DEFAULT_COMBINED_PERIODS = 'diachronic'

W2V_MODELS = {
	# 'COHA_byhalfcentury_nonf': {
	# 	"fn": "/Users/ryan/DH/keydata/data/db/models/COHA_byhalfcentury_nonf/chained_full_combined/1800-2000.min=100.run=01.txt",
	# 	'periods':['1800','1850','1900','1950'],
	# 	'periods_nice':['1800-1850','1850-1900','1900-1950','1950-2000'],
	# 	"corpus_desc":"COHA (Corpus of Historical American English), Non-Fiction"
	# },

	'COHA_byhalfcentury_nonf_smpl': {
		"fn": "models/COHA_byhalfcentury_nonf_smpl/model.txt",
		'periods':['1800','1850','1900','1950'],
		'periods_nice':['1800-1850','1850-1900','1900-1950','1950-2000'],
		"corpus_desc":"COHA (Corpus of Historical American English), Non-Fiction [Samples]"
	},

	# 'COHA_bythirtyyear_nonf': {
	# 	"fn": "/Users/ryan/DH/keydata/data/db/models/COHA_bythirtyyear_nonf_full/chained_combined/1810-2020.min=100.run=01.txt",
	# 	'periods':['1810','1840','1870','1900','1930','1960','1990'],
	# 	'periods_nice':['1810-1840','1840-1870','1870-1900','1900-1930','1930-1960','1960-1990','1990-2020'],
	# 	"corpus_desc":"COHA (Corpus of Historical American English), Non-Fiction"
	# },

	'ECCO_byquartercentury': {
		"fn": "models/ECCO_byquartercentury/model.txt.gz",
		'periods':['1700','1725','1750','1775'],
		'periods_nice':['1700-1725','1725-1750','1750-1775','1775-1800'],
		"corpus_desc":"ECCO (Eighteenth Century Collections Online)"
	}
}

DEFAULT_W2V_MODEL = 'COHA_byhalfcentury_nonf_smpl'
DEFAULT_W2V_FN = W2V_MODELS[DEFAULT_W2V_MODEL]['fn']
DEFAULT_PERIODS = W2V_MODELS[DEFAULT_W2V_MODEL]['periods']
DEFAULT_EXPAND_N = 2
DEFAULT_N_SIMILAR = 1
DEFAULT_N_TOP = 50





## DB
DB_NAME = 'keydata_toolbox'
DB_URL = 'mongodb://localhost:27017'
DB_NAMESPACE_VECS = 'vecs'
DB_NAMESPACE_DISTS = 'distnets/distnet'
DB_DIR = 'models/'



#MAX_NUM_VECS_TO_STORE = 100 
#MAX_NUM_VECS_TO_STORE = 50000

MAX_NUM_VECS_TO_STORE = 25000


