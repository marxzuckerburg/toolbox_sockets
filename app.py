# import
from app_config import *
from threading import Lock
from flask import Flask, render_template, session, request, copy_current_request_context
from flask_socketio import SocketIO, emit

# set up app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()


@app.route('/')
def analyze_word():
    params = {
      # 'W2V_MODELS': embed.W2V_MODELS,
      'DEFAULT_WORD_STR':DEFAULT_WORD_STR
    }
    return res.render('word.html',params)

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



  
@socketio.on('mostsimnet')
def mostsimnet(opts):
    msg='mostsimnet'
    log('starting '+msg+'()')
    print('mostsimnet_opts: ',opts)
    
    model = await embed.with_model(opts,log=log)

    //await model.build_vecdb(opts)
    //stop

    // opts['progress_range']=[0.5,0.75]
    progress(0.5, opts)
    most_similar_data = await model.get_most_similar(opts)
    progress(0.75, opts)
    print('most_similar_data',most_similar_data)
    //throw 'stop'
    

    //print('most_similar_data',most_similar_data)
    // opts['progress_range']=[0.75,0.9]
    // network_data = networks.sims2net(most_similar_data,opts)
    networks_data = networks.sims2net(most_similar_data,opts)
    progress(0.9, opts)

    log('finished '+msg+'()')
    // format response
    response_data = {'data':networks_data}
    io.to(socket.id).emit(msg+'_resp', response_data)
    progress(1.0,opts)
    // })
  })



  // EXPANDING WORDS
  socket.on('expandwords', function(opts) {
    var msg='expandwords'
    log('starting '+msg+'()', opts)
    embed.with_model(opts,log=log).then(function(model) {
      matches = model.get_expanded_wordset(opts)
      log('finished '+msg+'()')
      print('matches:',matches)

      io.to(socket.id).emit(msg+'_resp', matches)
    })
  })

  // UMAPPING 
  socket.on('get_umap', function(opts) {
    var msg='get_umap'
    log('starting '+msg+'()', opts)
    embed.with_model(opts,log=log).then(function(model) {
      log('periodizing input...')
      opts['words_orig']=opts['words']
      opts['words']=embed.periodize(opts['words'], opts['periods'])

      log('getting vectors for: '+opts['words'].join(', '))
      vector_data = model.get_vectors(opts)
      
      log('umapping data')
      umap_data = embed.get_umap_from_vector_data(vector_data)
      
      log('sending back to browser')
      io.to(socket.id).emit('get_umap_resp',umap_data);
    })
  });
})










// db=require('./db.js')
// db.test()









if __name__ == '__main__':
    app.run(debug=True)
