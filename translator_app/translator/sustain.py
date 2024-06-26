# here, we created a cache system so that
# once it is loaded in the memory, it can be used again and again
# same thing have not been done on model file because django can't deal with
# .h5 file in pikle format in memory as it usually does.
# alternative for this is, create session in django and keep using model untill
# session is alive.
from django.core.cache import cache
import pickle

model_cache_key = 'model_cache'
# this key is used to `set` and `get` your trained model from the cache

tokenizer = cache.get(model_cache_key) # get tokenizer from cache

if tokenizer is None:
    # your model isn't in the cache
    # so `set` it
    # load the tokenizer
    tokenizer = pickle.load(open('translator/tokenizer.pkl', 'rb'))
    cache.set(model_cache_key, tokenizer, None)
