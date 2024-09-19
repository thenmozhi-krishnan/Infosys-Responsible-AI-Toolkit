import uvicorn
from dotenv import load_dotenv
import os
from modeldeployment.config.logger import CustomLogger
from modeldeployment.config.config import read_config_yaml
from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from modeldeployment.routing.router import router
from fastapi.middleware.cors import CORSMiddleware

from aicloudlibs.utils.global_exception import UnSupportedMediaTypeException
from aicloudlibs.utils import global_exception_handler
load_dotenv()
log=CustomLogger()
app = FastAPI(**read_config_yaml('../config/metadata.yaml'))
"""

    Adding the CORS Middleware which handles the requests from different origins

    allow_origins - A list of origins that should be permitted to make cross-origin requests.
                    using ['*'] to allow any origin
    allow_methods - A list of HTTP methods that should be allowed for cross-origin requests.
                    using ['*'] to allow all standard method
    allow_headers - A list of HTTP request headers that should be supported for cross-origin requests. 
                    using ['*'] to allow all headers
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

"""
FAST API raise RequestValidationError in case request contains invalid data.
A global exception handler function to handle the requests which contains the invalid data

"""
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return  global_exception_handler.validation_error_handler(exc)


"""
A global exception handler function to handle the unsupported media type exception
"""
@app.exception_handler(UnSupportedMediaTypeException)
async def unsupported_mediatype_error_handler(request: Request, exc: UnSupportedMediaTypeException):
    return  global_exception_handler.unsupported_mediatype_error_handler(exc)



"""
A global exception handler function to handle the http exception
"""
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
     return  global_exception_handler.http_exception_handler(exc)



"""
incude the routing details of service
"""

app.include_router(router, prefix='/rai/v1', tags=["Infosys Responsible AI - LLM Moderation"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
    #,workers=int(os.getenv("WORKERS"))

