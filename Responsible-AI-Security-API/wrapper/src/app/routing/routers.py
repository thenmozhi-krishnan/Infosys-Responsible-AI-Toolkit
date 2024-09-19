'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


from fastapi import Depends,APIRouter,Query, Body,Form
from app.mappers.mappers import GetAttackDataRequest
from app.service.service import Infosys, Bulk
from app.config.logger import CustomLogger
from typing import Dict
import os
import gc
from app.util.auth.auth_client_id import get_auth_client_id
from app.util.auth.auth_jwt import get_auth_jwt
from app.util.auth.auth_none import get_auth_none
from fastapi import Depends, HTTPException, status
tool = APIRouter()
logs = APIRouter()
attack = APIRouter()
bulk = APIRouter()

log=CustomLogger()


auth_type=os.environ.get("AUTH_TYPE")
if auth_type=="azure":
    auth=get_auth_client_id()
elif auth_type=="jwt":
    auth=get_auth_jwt()
elif auth_type=="none":
    auth=get_auth_none()
else:
    raise HTTPException(status_code=500,detail="Invalid authentication type configured")
# ------------------------------------------------------------------------------------

@attack.post('/rai/v1/security_workbench/attack')
async def get_attacks(TargetClassifier:str=Form(), TargetDataType:str=Form(),auth=Depends(auth)):
    payload = {'targetClassifier':TargetClassifier, 'targetDataType':TargetDataType}
    response = Infosys.getAttackFuncs(payload)
    gc.collect()
    return response

@attack.post('/rai/v1/security_workbench/addattack')
def add_Attack(Payload:GetAttackDataRequest,auth=Depends(auth)):
    response = Infosys.addAttack(Payload)
    gc.collect()
    return response

@attack.delete('/rai/v1/security_workbench/deleteattack')
def delete_Attack(AttacFunc: str,auth=Depends(auth)):
    payload = {'attackName':AttacFunc}
    response = Infosys.deleteAttack(payload)
    gc.collect()
    return response

# ------------------------------------------------------------------------------------

@bulk.post('/rai/v1/security_workbench/runallattacks')
async def run_all_attacks(batchId:float=Form(...),auth=Depends(auth)):
    payload = {'batchid':batchId}
    response = Bulk.runAllAttack(payload)
    gc.collect()
    return {'BatchId':response}

# ------------------------------------------------------------------------------------


