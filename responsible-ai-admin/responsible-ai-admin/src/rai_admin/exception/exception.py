"""
# SPDX-License-Identifier: MIT
# Copyright 2024 - 2025 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
fileName: exception.py
description: handles usecase module specific exception
"""

import sys, traceback

from rai_admin.constants.local_constants  import SPACE_DELIMITER,PLACEHOLDER_TEXT,USECASE_ALREADY_EXISTS,USECASE_NOT_FOUND_ERROR,USECASE_NAME_VALIDATION_ERROR

from rai_admin.constants import local_constants as global_constants
from abc import ABC
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder


class RaiAdminException(Exception, ABC):
    """
    dscription: Abstract base class of UsecaseException.
    """

    def __init__(self, detail: str) -> None:
        self.status_code = global_constants.HTTP_STATUS_BAD_REQUEST
        super().__init__(detail)


class RaiAdminNotFoundError(RaiAdminException):
    """
    description: UsecaseNotFoundError thrown by usecase service
                 when the requested usecase details not found for a specific user.
    """
    def __init__(self,name):
        self.status_code = global_constants.HTTP_STATUS_NOT_FOUND
        self.detail =  USECASE_NOT_FOUND_ERROR.replace(PLACEHOLDER_TEXT,name)

class RaiAdminNameNotEmptyError(RaiAdminException):
    """
    description: UsecaseNameNotEmptyError thrown by create usecase service
                 when the requested usecase details not having usecase name.
    """
    def __init__(self,name):
        self.status_code = global_constants.HTTP_STATUS_409_CODE
        self.detail =  USECASE_NAME_VALIDATION_ERROR

class UnSupportedMediaTypeException(Exception):
    def __init__(self,contentTypeStr):
        self.status_code = global_constants.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        self.message = global_constants.UNSUPPPORTED_MEDIA_TYPE_ERROR + contentTypeStr
        
        
def validation_error_handler(exc: RequestValidationError):
    return JSONResponse(
        # status_code=int(constants.HTTP_422_UNPROCESSABLE_ENTITY),
        status_code=int(global_constants.HTTP_422_UNPROCESSABLE_ENTITY),
        content=jsonable_encoder({"detail": exc.errors()}),

    )

def  http_exception_handler(exc):
     return JSONResponse(
         status_code=exc.status_code,
         content=jsonable_encoder({"detail": str(exc.detail)})
    )

def unsupported_mediatype_error_handler(exc:UnSupportedMediaTypeException):
     return JSONResponse(
         status_code=exc.status_code,
         content=jsonable_encoder({"detail": str(exc.message)})
    )
