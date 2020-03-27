

VECS =  [
 'Animal (VG2) <> Human (VG2)',
 'Black <> White',
 'Concrete (CHPoetry-All) <> Abstract (CHPoetry-All)',
 'Concrete (CHPoetry-Robust) <> Abstract (CHPoetry-Robust)',
 'Concrete (Consolidated) <> Abstract (Consolidated)',
 'Concrete (ConsolidatedBinary) <> Abstract (ConsolidatedBinary)',
 'Concrete (HGI) <> Abstract (HGI)',
 'Derogatory (VG2) <> Non-Derogatory (VG2)]',
 'Female (HGI) <> Male (HGI)',
 'Female (VG2) <> Male (VG2)',
 'Hard Seed (RH&LL) <> Abstract Values (RH&LL)',
 'Negative (Abs-Cluster) <> Positive (Abs-Cluster)',
 'Negative (HGI) <> Positive (HGI)',
 'Object (VG2) <> Animal (VG2)',
 'Object (VG2) <> Animal+Human (VG2)',
 'Object (VG2) <> Human (VG2)',
 'Object (WN) <> Human (WN)',
 'Objective (Abs-Cluster) <> Subjective (Abs-Cluster)',
 'Passive (HGI) <> Active (HGI)',
 'Poetic Diction (CHPoetry) <> Prosaic Diction (CHPoetry)',
 'Pre-Norman (TU&JS) <> Post-Norman (TU&JS)',
 'Racialized (VG2) <> Non-Racialized (VG2)',
 'Science (Abs-Cluster)',
 'Submit (HGI) <> Power (HGI)',
 'Substance (Locke) <> Mode (Locke)',
 'Tangible (MT) <> Intangible (MT)',
 'The Natural (Abs-Cluster) <> The Social (Abs-Cluster)',
 'Vice (HGI) <> Virtue (HGI)',
 'Weak (HGI) <> Strong (HGI)',
 'Woman <> Man'];

FIELDS= ['Abstract(CHPoetry-All)',
 'Abstract(CHPoetry-Robust)',
 'Abstract(Consolidated)',
 'Abstract(ConsolidatedBinary)',
 'Abstract(HGI)',
 'AbstractValues(RH&LL)',
 'Active(HGI)',
 'Animal(VG2)',
 'Animal+Human(VG2)',
 'Black',
 'Concrete(CHPoetry-All)',
 'Concrete(CHPoetry-Robust)',
 'Concrete(Consolidated)',
 'Concrete(ConsolidatedBinary)',
 'Concrete(HGI)',
 'Derogatory(VG2)',
 'Female(HGI)',
 'Female(VG2)',
 'HardSeed(RH&LL)',
 'Human(VG2)',
 'Human(WN)',
 'Intangible(MT)',
 'Male(HGI)',
 'Male(VG2)',
 'Man',
 'Mode(Locke)',
 'Negative(Abs-Cluster)',
 'Negative(HGI)',
 'Non-Derogatory(VG2)]',
 'Non-Racialized(VG2)',
 'Object(VG2)',
 'Object(WN)',
 'Objective(Abs-Cluster)',
 'Passive(HGI)',
 'PoeticDiction(CHPoetry)',
 'Positive(Abs-Cluster)',
 'Positive(HGI)',
 'Post-Norman(TU&JS)',
 'Power(HGI)',
 'Pre-Norman(TU&JS)',
 'ProsaicDiction(CHPoetry)',
 'Racialized(VG2)',
 'Strong(HGI)',
 'Subjective(Abs-Cluster)',
 'Submit(HGI)',
 'Substance(Locke)',
 'Tangible(MT)',
 'TheNatural(Abs-Cluster)',
 'TheSocial(Abs-Cluster)',
 'Vice(HGI)',
 'Virtue(HGI)',
 'Weak(HGI)',
 'White',
 'Woman'];


GLOBAL_OPTS = {'fields':FIELDS, 'vecs':VECS}
GLOBAL_OPTS['all_fields_vecs'] = []
FIELDS.forEach(function(x) { GLOBAL_OPTS['all_fields_vecs'].push(x) })
VECS.forEach(function(x) { GLOBAL_OPTS['all_fields_vecs'].push(x) })
GLOBAL_OPTS['points']='movement'
GLOBAL_OPTS['view']='spaces'
GLOBAL_OPTS['x_vec_str']='King - Man + Woman'
GLOBAL_OPTS['y_vec_str']='Young - Old'


// console.log('GLOBAL_OPTS',GLOBAL_OPTS)











DEFAULT_W2V_FN = 'static/data/db/models/COHA_bythirtyyear_nonf_full/chained/1810-1840.txt'
