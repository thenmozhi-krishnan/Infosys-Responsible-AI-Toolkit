'''
MIT license https://opensource.org/licenses/MIT
Copyright 2024 Infosys Ltd
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from fastapi import HTTPException
from rai_backend.dao.Pageauthoritydb import PageauthorityDb
from rai_backend.dao.Pageauthoritynewdb import PageauthorityDbNew
from rai_backend.service.backend_service import UserInDB
from rai_backend.config.logger import CustomLogger
from rai_backend.dao.DatabaseConnection import DB
from rai_backend.dao.Userdb import UserDb
from rai_backend.domain.userEntity import User
from passlib.hash import pbkdf2_sha256

log = CustomLogger()
mydb = DB.connect()

class AuthService():
    mycol = mydb['User']

    @staticmethod
    def accountService(globalUsername):
        try:
            log.debug(globalUsername)
            return UserDb.getUserByName(globalUsername)
        except Exception as e:
            log.error("Error in Account Service", exc_info=True)
            raise HTTPException(status_code=401, detail="Unauthorized")

    @staticmethod
    def loginPostService(username, password):
        try:
            if not AuthService.validate_credentials(username, password):
                return "Invalid credentials", 400

          
            UserDb.add_initial_data()
            PageauthorityDb.add_initial_data()
            PageauthorityDbNew.add_initial_data()

            user = AuthService.mycol.find_one({'login': username}, {"_id": 0})
            if not user:
                return "User not found", 404

            if not user['activated']:
                return "User account is not activated", 403

            if not AuthService.verify_password(password, user['passwordHash']):
                return "Authentication failed", 401

            access_token = UserInDB.login_for_access_token(username)
            return {"id_token": access_token}
        except Exception as e:
            log.error("Error in Account Service", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    @staticmethod
    def signupPostService(newUser):
        try:
            if not AuthService.validate_new_user(newUser):
                return {"message": "Invalid user data", "status_code": 400}

            newUser['passwordHash'] = AuthService.get_password_hash(newUser['password'])
            del newUser['password']  

        
            UserDb.add_initial_data()
            success = UserDb.create(newUser)
            PageauthorityDb.add_initial_data()
            if success:
                return {"message": "User created successfully", "status_code": 200}
            else:
                return {"message": "User already exists", "status_code": 400}
        except Exception as e:
            log.error(f"Error in signup service: {str(e)}")
            return {"message": "An error occurred", "status_code": 500}

    @staticmethod
    def newUserRole(loginName, role):
        try:
            res = UserDb.update_newUser_role(loginName, role)
            return res
        except Exception as e:
            log.error(f"Error updating user role: {str(e)}")
            raise HTTPException(status_code=404, detail="Error updating user role")

    @staticmethod
    def resetCount():
        UserDb.myCount.update_one({},{'$set': {'counter':0}})

    def newAuthority(role):
       auth = UserDb.newAuthority(role)
       return auth


    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pbkdf2_sha256.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        return pbkdf2_sha256.hash(password)

    @staticmethod
    def validate_credentials(username: str, password: str) -> bool:
        return bool(username and password)

    @staticmethod
    def validate_new_user(user_data: dict) -> bool:
        required_fields = ['username', 'password', 'email']
        return all(field in user_data and user_data[field] for field in required_fields)