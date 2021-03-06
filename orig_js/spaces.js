











function get_custom_space(words = undefined,x_vec=undefined,x_vec_str=undefined,y_vec=undefined,y_vec_str=undefined) {

	var using_x_vec_str=false;
	var using_y_vec_str=false;
	if(x_vec_str!="undefined" & x_vec_str!=undefined & x_vec_str!="") {
		var using_x_vec_str=true;
		var x_col=x_vec_str;
	} else {
		var x_col=x_vec;
	}
	if(y_vec_str!="undefined" & y_vec_str!=undefined & y_vec_str!="") {
		var y_col=y_vec_str;
		var using_y_vec_str=true;
	} else {
		var y_col=y_vec;
	}

	word_str_need_vecs_for

	console.log('x_col',x_col);
	console.log('y_col',y_col);

	console.log('words',words)
	words_l=split_words(words.toLowerCase());
	console.log('words_l',words_l)
	dim_words_l=[x_col,y_col];

	words_l = words_l.filter(function(value, index, arr){ return value!="";});
	dim_words_l = dim_words_l.filter(function(value, index, arr){ return value!="";});

	console.log('words_l',words_l)
	console.log('dim_words_l',dim_words_l);


	all_vec_words=[]
	all_vec_words.push(...words_l)
	if(using_x_vec_str) { all_vec_words.push(...get_terms_from_formula(x_vec_str.toLowerCase())) }
	else { all_vec_words.push(x_vec) }
	if(using_y_vec_str) { all_vec_words.push(...get_terms_from_formula(y_vec_str.toLowerCase())) }
	else { all_vec_words.push(y_vec) }
	all_vec_words = all_vec_words.filter(function(value, index, arr){ return value!="" & value!="(" & value!=")";});
	console.log('<<all_vec_words>>',all_vec_words);


	// var AllVecs=[];
	var AllVecs={};
	var AllWords=[];
	promises = [];
	for(vi=0; vi<all_vec_words.length; vi++) {
		word=all_vec_words[vi];
		ifn='/static/data/db/matrices/COHA/'+word+'.tsv'
		promises.push(d3v5.tsv(ifn));
	}



Promise.all(promises).then(function(files) {
		var AllVecs={};
		var AllWords=[];
		var AllPeriods = [];

		$('#progressbar').html("")
		var bar = new ProgressBar.Circle(progressbar, {
			color: '#aaa',
			// This has to be the same size as the maximum width to
			// prevent clipping
			strokeWidth: 6,
			trailWidth: 1,
			// easing: 'easeInOut',
			duration: 0,
			text: {
				autoStyleContainer: false
			},
			from: { color: '#aaa', width: 1 },
			to: { color: '#333', width: 4 },
			// Set default step function for all animate calls
			step: function(state, circle) {
				circle.path.setAttribute('stroke', state.color);
				circle.path.setAttribute('stroke-width', state.width);

				var value = Math.round(circle.value() * 100);
				if (value === 0) {
					circle.setText('');
				} else {
					circle.setText(value+'%');
				}

			}
		});
		// bar.text.style.fontFamily = '"Raleway", Helvetica, sans-serif';
		bar.text.style.fontSize = '1rem';
		bar.animate(0.0);

		console.log('bar',bar);

		for(fi=0;fi<files.length;fi++) {
			fdata=files[fi];
			vec_word=all_vec_words[fi];

			for(fdi=0; fdi<fdata.length;fdi++) {
				// add word_period
				dat=fdata[fdi];
				Word=dat['word']+'_'+dat['period'];
				AllWords.push(Word);

				// add word_period vec
				vecstr=dat['vectors'];
				vecfl=parse_nums(vecstr);
				AllVecs[Word.replace(' ','')]=vecfl;

				// add period?
				if (!AllPeriods.includes(dat['period'])) {
					AllPeriods.push(dat['period']);
				}
			}

		bar.animate((fi+1)/files.length);
		}
		console.log('AllVecs',words_l);
		console.log('AllWords',AllPeriods);
		console.log('AllPeriods',AllPeriods);




		// get data to output
		sim_ld = [];
		// all_vals = [];
		x_vals=[];
		y_vals=[];

		words_l.forEach(function(row_word) {
			AllPeriods.forEach(function(period) {
				ok_word=true;
				oword_dx = {'word':row_word, 'period':period}



				dim_words_l.forEach(function(dim_word) {
					// console.log(period,row_word,dim_word,'.....');
					row_word_period = row_word+'_'+period;
					dim_word_period = dim_word+'_'+period;

					console.log('dim_word:',dim_word)
					console.log('using_x_vec_str:',using_x_vec_str)
					console.log('x_col:',x_col)
					console.log('using_y_vec_str:',using_y_vec_str)
					console.log('y_col:',y_col)

					if(using_y_vec_str & dim_word==y_col) {
						dim_word_period_vec = parse_formula(dim_word.toLowerCase(),AllVecs,suffix='_'+period)
						console.log('dim_vec Ycol formula result:',dim_word,dim_word_period_vec)
					} else if(using_x_vec_str & dim_word==x_col) {
						dim_word_period_vec = parse_formula(dim_word.toLowerCase(),AllVecs,suffix='_'+period)
						console.log('dim_vec Xcol formula result:',dim_word,dim_word_period_vec)
					} else {
						dim_word_period_vec = AllVecs[dim_word_period.replace(' ','')];

					}
					row_word_period_vec = AllVecs[row_word_period];






					console.log('VECS:',[row_word_period,dim_word_period],[row_word_period_vec,dim_word_period_vec]);

					if (row_word_period_vec!=undefined & dim_word_period_vec!=undefined) {
								var csim_val = similarity(row_word_period_vec,dim_word_period_vec)

								// invert if formula-fied
								if(dim_word.includes('+') | dim_word.includes('-') | dim_word.includes('*') | dim_word.includes('/')) {
								// if(tokenize(dim_word).length>1) {
								// console.log('INVERTING:',dim_word,csim_val)
									// var csim_val = -1 * csim_val
									csim_val=csim_val;
								}


						// all_vals.push(csim_val);
						if(dim_word==y_col) { y_vals.push(csim_val) }
						if(dim_word==x_col) { x_vals.push(csim_val) }

						oword_dx[dim_word]= csim_val

						// console.log('CSIM!',csim_val);
					} else {
						ok_word=false;
					}


				});

			if(ok_word){ sim_ld.push(oword_dx); }
			// sim_ld.push(oword_dx);
			// console.log('oword_dx',oword_dx)
			});
		});

			attached_data = sim_ld;

			console.log('attached_data000',attached_data)
			input_words=words_l.join(',')



			console.log('y_vals_length',y_vals.length,y_vals)
			console.log('x_vals_length',x_vals.length,x_vals)
			y_min=Math.min(...y_vals)
			y_max=Math.max(...y_vals)
			x_min=Math.min(...x_vals)
			x_max=Math.max(...x_vals)

			y_margin=(y_max-y_min)/10
			x_margin=(x_max-x_min)/10

			y_mid=y_min + (y_max - y_min)/2
			x_mid=x_min + (x_max - x_min)/2

			all_periods = use_all_periods()

			if(using_y_vec_str){ y_col_name='V('+tokenize(y_col).join(' ')+')'; } else { y_col_name=y_col; }
			if(using_x_vec_str){ x_col_name='V('+tokenize(x_col).join(' ')+')'; } else { x_col_name=x_col; }

			make_linegraph_spaces(
				input_words,
				y_col = y_col,
				x_col = x_col,
				div_id="custom_viz",
				x_min=x_min-x_margin, x_max=x_max+x_margin,
				y_min=y_min-y_margin,y_max=y_max+y_margin,
				y_mid=y_mid,x_mid=x_mid,
				sp_title="",
				y_col_name = y_col_name, //"<< Concrete | Abstract >>",
				x_col_name = x_col_name,
				all_periods=all_periods, orig_width=600, orig_height=600,
				attached_data = attached_data,ifn_dir=IFN_DIR,words=words_l); //,words=sim_ld_words);

	})

}






// exports.get_umap_from_vector_data=get_umap_from_vector_data