'''
Copyright 2024-2025 Infosys Ltd.
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import shutil
import pymongo
import datetime,time
from docProcess.dao.DatabaseConnection import DB
from dotenv import load_dotenv
from docProcess.config.logger import CustomLogger
from gridfs import GridFS
from bson import ObjectId
from fastapi.responses import StreamingResponse
import io
import os

load_dotenv()
log = CustomLogger()

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


mydb=DB.connect()


class fileStoreDb:

    fs=GridFS(mydb)

    def findOne(id):
        file=fileStoreDb.fs.find_one({"_id":id})
        if file:
            content= file.read()
            return {"fileName": file.filename, "contentType": file.contentType, "data": content}
        # values=fileStoreDb.mycol.find({"_id":id},{})[0]
        # print(values)
        values=AttributeDict(values)
        return values
    
    def get_file_stream(file_id):
        try:
            file = fileStoreDb.fs.get(ObjectId(file_id))
            file_stream = io.BytesIO(file.read())
            # Create streaming response
            return StreamingResponse(
                file_stream,
                media_type=file.content_type or 'application/octet-stream',
                headers={
                    "Content-Disposition": f'attachment; filename="{file.filename}"'
                }
            )
        except Exception as e:
            log.error(e.__dict__)
            log.info("Issue in get_file_stream()")  

    def get_download_link_with_base_url(file_id: str, base_url: str) -> str:
        return f"{base_url}/download/{file_id}"    

    def findall(query):
        value_list=[]
        values=fileStoreDb.mycol.find(query,{})
        for v in values:

            v=AttributeDict(v)
            value_list.append(v)
        return value_list
    
    def create(value):
         print("Starting file storage process")
         value=AttributeDict(value)
         fileid:any
         with fileStoreDb.fs.new_file(filename=value.fileName,content_type=value.type) as f:
             shutil.copyfileobj(value.file,f)
             fileid=f._id
           
         return fileid

    
    def update(id,value:dict):
      
        newvalues = { "$set": value }
        docProccessData=fileStoreDb.mycol.update_one({"_id":id},newvalues)
        log.debug(str(newvalues)) 
        return docProccessData.acknowledged
    
    def delete(id):
        fileStoreDb.mycol.delete_many({"_id":id})
        # DocProcDtl.mycol.delete_many({})
        # Docpagedtl.mycol.delete_many({})