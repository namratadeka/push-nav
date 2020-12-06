import pybullet as p
import pybullet_data
import os


class Plane:
    def __init__(self, client):
        # f_name = os.path.join(os.path.dirname(__file__), 'simpleplane.urdf')
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(fileName='plane.urdf',
                   physicsClientId=client)
