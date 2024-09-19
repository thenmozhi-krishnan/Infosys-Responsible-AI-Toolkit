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
