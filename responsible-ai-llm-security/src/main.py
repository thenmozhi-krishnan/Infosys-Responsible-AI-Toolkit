'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from fastapi import FastAPI
from routing.routing import model,robustness,datasets,results
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config.config import read_config_yaml
from starlette.middleware.base import BaseHTTPMiddleware 
import os

origins =os.getenv("ALLOW_ORIGINS")

class CacheControlMiddleware(BaseHTTPMiddleware): 
    async def dispatch(self, request, call_next): 
        response = await call_next(request) 
        response.headers["Cache-Control"] = "private, no-store" 
        return response 

class XSSProtectionMiddleware(BaseHTTPMiddleware): 
    async def dispatch(self, request, call_next): 
        response = await call_next(request) 
        response.headers['X-XSS-Protection'] = '1; mode=block' 
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src data: https:; object-src 'none'; script-src https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js 'self' 'unsafe-inline';style-src https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css 'self' 'unsafe-inline'; upgrade-insecure-requests;"
        return response 

 
app = FastAPI(**read_config_yaml('../config/metadata.yaml'))
app.add_middleware(CacheControlMiddleware)
app.add_middleware(XSSProtectionMiddleware) 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origins],
    allow_methods=["POST","GET"],
    allow_headers=["*"],
)

app.include_router(model,tags=["LLM Models"])
app.include_router(datasets,tags=["LLM Datasets"])
app.include_router(robustness,tags=["LLM Robustness"])
app.include_router(results,tags=["LLM ExternalResults"])
if __name__== "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=80)
    #uvicorn.run("main:app", reload=True, port=30040)