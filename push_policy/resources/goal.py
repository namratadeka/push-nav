import pybullet as p
import os


class Goal:
    def __init__(self, client, base, test=False):
        f_name = os.path.join(os.path.dirname(__file__), 'simplegoal.urdf')
        if test:
        	self.id = p.loadURDF(fileName=f_name,
        						 basePosition=[5, 0, 0],
        						 physicsClientId=client)
        else:
	        self.id = p.loadURDF(fileName=f_name,
	                   basePosition=[base[0], base[1], 0],
	                   physicsClientId=client)
