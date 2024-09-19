"""
fileName: exception.py
description: handles usecase module specific exception
"""

import sys, traceback

from modeldeployment.constants.local_constants  import SPACE_DELIMITER,PLACEHOLDER_TEXT,USECASE_ALREADY_EXISTS,USECASE_NOT_FOUND_ERROR,USECASE_NAME_VALIDATION_ERROR

from aicloudlibs.constants import constants as global_constants
from abc import ABC


class modeldeploymentException(Exception, ABC):
    """
    dscription: Abstract base class of UsecaseException.
    """

    def __init__(self, detail: str) -> None:
        self.status_code = global_constants.HTTP_STATUS_BAD_REQUEST
        super().__init__(detail)


class modeldeploymentNotFoundError(modeldeploymentException):
    """
    description: UsecaseNotFoundError thrown by usecase service
                 when the requested usecase details not found for a specific user.
    """
    def __init__(self,name):
        self.status_code = global_constants.HTTP_STATUS_NOT_FOUND
        self.detail =  USECASE_NOT_FOUND_ERROR.replace(PLACEHOLDER_TEXT,name)

class modeldeploymentNameNotEmptyError(modeldeploymentException):
    """
    description: UsecaseNameNotEmptyError thrown by create usecase service
                 when the requested usecase details not having usecase name.
    """
    def __init__(self,name):
        self.status_code = global_constants.HTTP_STATUS_409_CODE
        self.detail =  USECASE_NAME_VALIDATION_ERROR
