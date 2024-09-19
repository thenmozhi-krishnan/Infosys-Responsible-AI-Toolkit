import time
import hashlib
from config.logger import CustomLogger


log = CustomLogger()


################################# Caching Logic ######################################
# LRU Caching Technique
'''
   ttl : time for which cache entries can be stored

   max_size : maximum number of cache entries that can be stored , here each entry is a key-value pair.
              Key is a prompt and value is the response for that prompt. So, if max_size is 1000, then it
              means that we can store 1000 prompt-response pairs in cache.
'''
def lru_cache_response(func,ttl=7200,max_size=1000):
    cache = {}
    lru_queue = []
    
    def generate_md5_hash(input_string):
        encoded_string = input_string.encode()
        md5_hash = hashlib.md5(encoded_string)
        hex_digest = md5_hash.hexdigest()
        return hex_digest

    def wrapper(*args, **kwargs):
        key = generate_md5_hash(str(args) + str(kwargs))
        now = time.time()

        if key in cache and now - cache[key][1] < ttl:
            log.info("Within time limit !!!")
            lru_queue.remove(key)
            lru_queue.append(key)
            return cache[key][0]
        
        if len(cache) > max_size:
            log.info("Cache is full, remove oldest prompt and its response")
            oldest_key = lru_queue.pop(0)
            del cache[oldest_key]

        log.info("May be Time exceeded or memory is full or it's a new entry for cache !!")
        result = func(*args, **kwargs)
        cache[key] = (result, now)
        lru_queue.append(key)
        return result
    
    return wrapper


def lru_cache(ttl,size):
    def decorator(func):
        return lru_cache_response(func,ttl,size)
    return decorator
