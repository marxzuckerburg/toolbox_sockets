// config
// DEFAULT_W2V_FN = 'static/data/db/models/COHA_bythirtyyear_nonf_full/chained/1810-1840.txt'
// DEFAULT_W2V_FN = 'static/data/db/models/COHA_bythirtyyear_nonf_full/chained/1990-2020.txt'
//DEFAULT_W2V_FN = 'static/data/db/models/COHA_bythirtyyear_nonf_full/chained_combined/1810-2020.min=100.run=01.txt'
DEFAULT_W2V_FN="/Volumes/Present/DH/data/models/COHA_byhalfcentury_nonf/chained_combined/1800-1999.min=500.run=01.txt"
// DEFAULT_W2V_FN="/Volumes/Present/DH/data/models/COHA_byhalfcentury_nonf/chained_full_combined/1800-2000.min=100.run=01.txt"
// DEFAULT_W2V_FN="/Volumes/Present/DH/data/models/COHA_byhalfcentury_nonf/separate3_allskips/1800-1849.txt.run=01.txt"
DEFAULT_WORD_STR="value,price,importance,value-price,value-importance"
DB_CONN_STR='redis://localhost:6379'


VECS = ['Animal (VG2) <> Human (VG2)', 'Black <> White', 'Concrete (CHPoetry-All) <> Abstract (CHPoetry-All)', 'Concrete (CHPoetry-Robust) <> Abstract (CHPoetry-Robust)', 'Concrete (Consolidated) <> Abstract (Consolidated)', 'Concrete (ConsolidatedBinary) <> Abstract (ConsolidatedBinary)', 'Concrete (HGI) <> Abstract (HGI)', 'Derogatory (VG2) <> Non-Derogatory (VG2)]', 'Female (HGI) <> Male (HGI)', 'Female (VG2) <> Male (VG2)', 'Hard Seed (RH&LL) <> Abstract Values (RH&LL)', 'Negative (Abs-Cluster) <> Positive (Abs-Cluster)', 'Negative (HGI) <> Positive (HGI)', 'Object (VG2) <> Animal (VG2)', 'Object (VG2) <> Animal+Human (VG2)', 'Object (VG2) <> Human (VG2)', 'Object (WN) <> Human (WN)', 'Objective (Abs-Cluster) <> Subjective (Abs-Cluster)', 'Passive (HGI) <> Active (HGI)', 'Poetic Diction (CHPoetry) <> Prosaic Diction (CHPoetry)', 'Pre-Norman (TU&JS) <> Post-Norman (TU&JS)', 'Racialized (VG2) <> Non-Racialized (VG2)', 'Science (Abs-Cluster)', 'Submit (HGI) <> Power (HGI)', 'Substance (Locke) <> Mode (Locke)', 'Tangible (MT) <> Intangible (MT)', 'The Natural (Abs-Cluster) <> The Social (Abs-Cluster)', 'Vice (HGI) <> Virtue (HGI)', 'Weak (HGI) <> Strong (HGI)', 'Woman <> Man']
FIELDS = ['Abstract(CHPoetry-All)', 'Abstract(CHPoetry-Robust)', 'Abstract(Consolidated)', 'Abstract(ConsolidatedBinary)', 'Abstract(HGI)', 'AbstractValues(RH&LL)', 'Active(HGI)', 'Animal(VG2)', 'Animal+Human(VG2)', 'Black', 'Concrete(CHPoetry-All)', 'Concrete(CHPoetry-Robust)', 'Concrete(Consolidated)', 'Concrete(ConsolidatedBinary)', 'Concrete(HGI)', 'Derogatory(VG2)', 'Female(HGI)', 'Female(VG2)', 'HardSeed(RH&LL)', 'Human(VG2)', 'Human(WN)', 'Intangible(MT)', 'Male(HGI)', 'Male(VG2)', 'Man', 'Mode(Locke)', 'Negative(Abs-Cluster)', 'Negative(HGI)', 'Non-Derogatory(VG2)]', 'Non-Racialized(VG2)', 'Object(VG2)', 'Object(WN)', 'Objective(Abs-Cluster)', 'Passive(HGI)', 'PoeticDiction(CHPoetry)', 'Positive(Abs-Cluster)', 'Positive(HGI)', 'Post-Norman(TU&JS)', 'Power(HGI)', 'Pre-Norman(TU&JS)', 'ProsaicDiction(CHPoetry)', 'Racialized(VG2)', 'Strong(HGI)', 'Subjective(Abs-Cluster)', 'Submit(HGI)', 'Substance(Locke)', 'Tangible(MT)', 'TheNatural(Abs-Cluster)', 'TheSocial(Abs-Cluster)', 'Vice(HGI)', 'Virtue(HGI)', 'Weak(HGI)', 'White', 'Woman']
DEFAULT_N_SIMILAR = 10
// DEFAULT_CSIM_CUTOFF = 0.45
DEFAULT_CSIM_CUTOFF = undefined   // ie leave it at max DEFAULT_N_SIMILAR

// store in opts
GLOBAL_OPTS = {'fields':FIELDS, 'vecs':VECS}
GLOBAL_OPTS['all_fields_vecs'] = []
FIELDS.forEach(function(x) { GLOBAL_OPTS['all_fields_vecs'].push(x) })
VECS.forEach(function(x) { GLOBAL_OPTS['all_fields_vecs'].push(x) })
GLOBAL_OPTS['points']='movement'
GLOBAL_OPTS['view']='spaces'
GLOBAL_OPTS['x_vec_str']='King - Man + Woman'
GLOBAL_OPTS['y_vec_str']='Young - Old'

// start server
const express = require('express');
const app = express();
const port = 30101;
var http = require('http').createServer(app);
var io = require('socket.io')(http);
app.use('/static', express.static('static'))
var nunjucks = require('nunjucks')
nunjucks.configure('templates/', { autoescape: false, express: app });

// const pd = require('pandas-js')

function shuffle(array) {
  var currentIndex = array.length, temporaryValue, randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}



// Word2vec
var embed = require('./embedding.js')
var networks = require('./networks.js')
var spaces = require('./spaces.js')


// http routing
app.get('/', function(req, res){ 
  // console.log(embed.W2V_MODEL_FNS)

  // embed.with_model().then(function(model) {
    params = {
      // w2v_fns:embed.W2V_MODEL_FNS,
      // input_options:options,
      W2V_MODELS: embed.W2V_MODELS,
      DEFAULT_WORD_STR:DEFAULT_WORD_STR
    }
    return res.render('word.html',params);   
  // })
});

http.listen(port, function(){ console.log('listening on *:'+port); });

function norm_val_within_range(domain_val, range=[0.0,1.0], domain=[0.0,1.0]) {
  dval = (domain_val-domain[0]) / (domain[1]-domain[0])
  dval_in_range = (dval * (range[1]-range[0])) + range[0]
  return dval_in_range
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}   


// SOCKET ROUTING
var num_conn=0

io.on('connection', function(socket){
  num_conn+=1
  var log = function(x) { console.log(x); io.to(socket.id).emit('status','(server) '+x); }
  

  // embed.with_model(opts,log=log).then(function(model) {
  //     opts['word']=['value_1950-importance_1950']
  //     model.get_vector(opts)
  //     throw 'testing!' 
  // })

  // log(num_conn.toString() + ' ppl connected')

  var progress = function(domain_val,opts) { 
    range=opts['progress_range']
    domain=opts['progress_domain']
    if(range==undefined) { range=[0,1] }
    if(domain==undefined) { domain = [0,1] }
    dval_in_range = norm_val_within_range(domain_val,range=range,domain=domain)
    // console.log('progress!',domain_val,range,domain,'-->',dval_in_range)
    // console.log('PROGRESS:',dval_in_range*100,'%'); 
    // io.to(socket.id).emit('progress',dval_in_range);
  }
  
  
  // Most similar
  socket.on('mostsim', function(opts) {
    var msg='mostsim'
    log('starting '+msg+'()')
    embed.with_model(opts,log=log,progress=progress).then(function(model) {
      most_similar_data = model.get_most_similar(opts) 
      log('finished '+msg+'()')
      io.to(socket.id).emit(msg+'_resp', most_similar_data)
    })
  })

  // Draw most simialar
  
  socket.on('mostsimnet', async function(opts) {
    var msg='mostsimnet'
    log('starting '+msg+'()')

    console.log('mostsimnet_opts: ',opts)
    progress(0.25, opts)

    model = await embed.with_model(opts,log=log)

    //await model.build_vecdb(opts)
    //stop

    // opts['progress_range']=[0.5,0.75]
    progress(0.5, opts)
    most_similar_data = await model.get_most_similar(opts)
    progress(0.75, opts)
    console.log('most_similar_data',most_similar_data)
    //throw 'stop'
    

    //console.log('most_similar_data',most_similar_data)
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
      console.log('matches:',matches)

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
