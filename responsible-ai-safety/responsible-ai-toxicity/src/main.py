"""
 <2023> Infosys Limited, Bangalore, India. All Rights Reserved.
 Version: 
Except for any free or open source software components embedded in this Infosys proprietary software program ( Program ), 
this Program is protected by copyright laws, international treaties and other pending or existing intellectual property rights in India, 
the United States and other countries. Except as expressly permitted, any unauthorized reproduction, storage, 
transmission in any form or by any means (including without limitation electronic, mechanical, printing, photocopying, recording or otherwise), 
or any distribution of this Program, or any portion of it, may result in severe civil and criminal penalties, 
and will be prosecuted to the maximum extent possible under the law.
"""

"""

app: Project Management service 
fileName: main.py
description: Project management services helps to create Usecase and projects .
             This app handles the services for usecase module which perform CRUD operaions.

"""
from typing import List


import uvicorn
from profanity.config.logger import CustomLogger
from fastapi import Depends, FastAPI,  Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from profanity.routing.profanity_router import router
from fastapi.middleware.cors import CORSMiddleware

from aicloudlibs.utils.global_exception import UnSupportedMediaTypeException
from aicloudlibs.utils import global_exception_handler


log=CustomLogger()
## initialize the app with openapi and docs url
app = FastAPI(openapi_url="/api/v1/safety/openapi.json", docs_url="/api/v1/safety/docs")

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
app.include_router(router, prefix='/api/v1', tags=['Infosys Responsible AI - Profanity'])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)



