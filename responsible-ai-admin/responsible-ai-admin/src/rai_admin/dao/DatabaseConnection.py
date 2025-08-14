"""
# SPDX-License-Identifier: MIT
# Copyright 2024 - 2025 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import pymongo

from dotenv import load_dotenv
from rai_admin.config.logger import CustomLogger
import sys
load_dotenv()

log = CustomLogger()


class DB:
    def connect():
        try:
            db_type = os.getenv('DB_TYPE', 'mongo').lower()

            # Connect to the appropriate database based on the database type
            if db_type == 'cosmos':
                myclient = pymongo.MongoClient(os.getenv("COSMOS_PATH"))
            elif db_type == 'mongo':
                myclient = pymongo.MongoClient(os.getenv("MONGO_PATH"))
            # Add more elif conditions for new databases here
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            mydb = myclient[os.getenv("DB_NAME")]
            return mydb
        except Exception as e:
            log.info(str(e))
            sys.exit()
            
