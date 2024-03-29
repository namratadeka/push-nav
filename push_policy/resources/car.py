import pybullet as p
import os
import math
import numpy as np 


class Car:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'car.urdf')
        self.id = p.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0.1],
                              physicsClientId=client)

        # Joint indices as found by p.getJointInfo()
        self.head_joint = 0
        p.enableJointForceTorqueSensor(self.id, self.head_joint, True)
        self.steering_joints = [1, 3]
        self.drive_joints = [2, 4, 5, 6]
        # Joint speed
        self.joint_speed = 0
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant increases "speed" of the car
        self.c_throttle = 20

    def get_ids(self):
        return self.id, self.client

    def apply_action(self, action):
        # Expects action to be two dimensional
        throttle, steering_angle = action

        # Clip steering angle to reasonable values
        # steering_angle = max(min(steering_angle, 0.6), -0.6)

        # Set the steering joint positions
        p.setJointMotorControlArray(self.id, self.steering_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[steering_angle] * 2,
                                    physicsClientId=self.client)

        # Calculate drag / mechanical resistance ourselves
        # Using velocity control, as torque control requires precise models
        friction = -self.joint_speed * (self.joint_speed * self.c_drag +
                                        self.c_rolling)
        acceleration = self.c_throttle * throttle + friction
        # Each time step is 1/240 of a second
        self.joint_speed = self.joint_speed + 1/30 * acceleration
        self.joint_speed = max(-1, self.joint_speed)

        # Set the velocity of the wheel joints directly
        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.joint_speed] * 4,
            forces=[2.4] * 4,
            physicsClientId=self.client)

    def get_camera_image(self):
        image = np.zeros((100, 100, 4))
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(self.id, self.client)]
        pos[2] = 0.3
        pos[0] += 0.25

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        w, h, rgb, depth, self.segmask = p.getCameraImage(84, 84, view_matrix, proj_matrix)
        rgb = rgb[:, :, :-1] / 255.
        rgbd = np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2)

        return rgbd

    def get_observation(self):
        # Get the position and orientation of the car in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.id, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]
        # Get the velocity of the car
        vel = p.getBaseVelocity(self.id, self.client)[0][0:2]

        # Concatenate position, orientation, velocity
        observation = (pos + ori + vel)

        # Get contact forces of obstacles with the base & head link
        base_collision_force = np.zeros(3)
        head_reaction_force = tuple(np.zeros(6).tolist())

        contact_points = p.getContactPoints(bodyA=self.id)
        for contact_info in contact_points:
            bodyB = contact_info[2]
            bodyA_index = contact_info[3]
            if bodyB != self.id:
                if bodyA_index == -1:  # collision with base
                    contact_force = contact_info[9]
                    contact_normal = contact_info[7]
                    base_collision_force += np.array(contact_normal) * contact_force
                if bodyA_index == 0:   # collision with head
                    head_reaction_force = p.getJointState(self.id, self.head_joint)[2]

        self.head_force = np.linalg.norm(np.array(head_reaction_force[:3]))

        observation += tuple(base_collision_force.tolist())
        observation += head_reaction_force
        rgbd = self.get_camera_image()

        return observation, rgbd
        