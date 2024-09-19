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

from configparser import ConfigParser
import yaml


def readConfig(section,filename):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


def read_config_yaml(filename):
    with open(filename) as config_file:
        config_details = yaml.safe_load(config_file)
    return config_details
