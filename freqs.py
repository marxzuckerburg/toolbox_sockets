import numpy as np
from models import *
import pandas as pd

fn2meta = {}
fn2sums = {}



def get_df_sums(hdf_fn, total_key='_totals'):
	if not hdf_fn in fn2sums:
		print('>> loading freq sums...')
		df_meta=get_df_meta(hdf_fn)
		print('meta cols:',df_meta.columns)
		fn2sums[hdf_fn]=df_meta.groupby(['year','genre']).sum().fillna(0)
	return fn2sums[hdf_fn]

def get_df_meta(hdf_fn, meta_key='_metadata'):
	if not hdf_fn in fn2meta:
		print('>> loading freq metadata...')
		df_meta=pd.read_hdf(hdf_fn, meta_key).fillna(0)

		# hack?
		if 'ECCO' in hdf_fn: df_meta=df_meta.query('1700 <= year <= 1799')

		df_meta['num_words']=df_meta['num_words'].fillna(0).apply(int)
		df_meta['num_content_words']=df_meta['num_content_words'].fillna(0).apply(int)

		fn2meta[hdf_fn]=df_meta
	return fn2meta[hdf_fn]


def get_df_freqs(word, model_opts, moving_average_window=25, min_periods=10):
	hdf_fn_meta = model_opts['fn_meta']
	hdf_fn = model_opts['fn_freqs']

	df_meta = get_df_meta(hdf_fn_meta)
	df_sums = get_df_sums(hdf_fn_meta)
	
	try:
		df_word = pd.read_hdf(hdf_fn, word).fillna(0)
	except KeyError:
		return None


	df_word_meta = df_word.join(df_meta)
	df_word_meta_totals = df_word_meta.groupby(['year','genre']).sum().reset_index()
	df_word_meta_totals = df_word_meta_totals.merge(df_sums, on=['year','genre'], suffixes=['','_total'])

	df_word_meta_totals['freq']=[x/y for x,y in zip(df_word_meta_totals['count'], df_word_meta_totals['num_words_total'])]
	df_word_meta_totals['freq_content']=[x/y for x,y in zip(df_word_meta_totals['count'], df_word_meta_totals['num_content_words_total'])]
	
	# return crosstab version
	df_res=df_word_meta_totals
	df_tabs=pd.crosstab(df_res.year, df_res.genre, df_res.freq_content, aggfunc=np.mean).fillna(0).reset_index()
	df_tabs.columns=[('date' if i==0 else x) for i,x in enumerate(df_tabs.columns)]

	# smooth?
	df_tabs = df_tabs.set_index('date').rolling(window=moving_average_window, min_periods=min_periods).mean().reset_index().iloc[min_periods:]

	return df_tabs