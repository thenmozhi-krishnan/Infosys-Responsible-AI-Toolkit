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
fileName: exception.py
description: handles usecase module specific exception
"""

import sys, traceback

from profanity.constants.local_constants  import SPACE_DELIMITER,PLACEHOLDER_TEXT,USECASE_ALREADY_EXISTS,USECASE_NOT_FOUND_ERROR,USECASE_NAME_VALIDATION_ERROR

from aicloudlibs.constants import constants as global_constants
from abc import ABC


class ProfanityException(Exception, ABC):
    """
    dscription: Abstract base class of UsecaseException.
    """

    def __init__(self, detail: str) -> None:
        self.status_code = global_constants.HTTP_STATUS_BAD_REQUEST
        super().__init__(detail)


class ProfanityNotFoundError(ProfanityException):
    """
    description: UsecaseNotFoundError thrown by usecase service
                 when the requested usecase details not found for a specific user.
    """
    def __init__(self,name):
        self.status_code = global_constants.HTTP_STATUS_NOT_FOUND
        self.detail =  USECASE_NOT_FOUND_ERROR.replace(PLACEHOLDER_TEXT,name)

class ProfanityNameNotEmptyError(ProfanityException):
    """
    description: UsecaseNameNotEmptyError thrown by create usecase service
                 when the requested usecase details not having usecase name.
    """
    def __init__(self,name):
        self.status_code = global_constants.HTTP_STATUS_409_CODE
        self.detail =  USECASE_NAME_VALIDATION_ERROR
