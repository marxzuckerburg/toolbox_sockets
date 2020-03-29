# import globals
from threading import Lock
from flask import Flask, render_template, session, request, copy_current_request_context
from flask_socketio import SocketIO, emit

# import our own code
from app_config import *
from networks import *
from models import *
from db import *

# set up app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

# logger
def log(x):
		print(x)
		emit('status','(server) '+x)

id2M={}


@app.route('/')
def analyze_word():
		return render_template('word.html',DEFAULT_WORD_STR=DEFAULT_WORD_STR,W2V_MODELS=W2V_MODELS)

@app.route('/about')
def index():
	return render_template('about.html')

@app.route('/corpus_coha')
def corpus_coha():
	return render_template('corpus_coha.html')

@app.route('/manifestos')
def manifestos():
		return render_template('manifestos.html')

@app.route('/manifestos_feb22')
def manifestos_feb22():
		return render_template('manifestos_feb22.html')

def get_model(opts):
	global id2M

	Mid=opts['model_id']
	if Mid in id2M:
		model = id2M[Mid]
	else:
		model = id2M[Mid] = Embedding(id=Mid,fn=opts['model_fn'],periods=opts['model_periods'])
	return model

	
@socketio.on('mostsimnet')
def mostsimnet(opts):
	msg='mostsimnet'
	log('starting '+msg+'()')
	print('mostsimnet_opts: ',opts)
	
	log('getting model')
	model=get_model(opts)

	log('getting sim data')
	try:
		n_top = int(opts['n_top'])
	except ValueError:
		n_top = DEFAULT_N_TOP

	most_similar_data = model.get_most_similar(words=opts['words'],n_top=n_top,periods=opts['periods'],combine_periods=opts['combine_periods'])
	
	log('making network')
	networks_data = sims2net(most_similar_data,combine_periods=opts['combine_periods'])

	emit(msg+'_resp', {'data':networks_data})
	


if __name__ == '__main__':
	# app.run(debug=True)
	import sys
	print(sys.argv)
	port=1799
	if len(sys.argv)>1:
		port=int(sys.argv[-1])
	print(f' * Starting server on port: {port}')
	socketio.run(app, debug=True, port=port,host='0.0.0.0')