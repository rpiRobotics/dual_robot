import numpy as np
from general_robotics_toolbox import *
import traceback, time, copy
from qpsolvers import solve_qp
from scipy.optimize import differential_evolution


###Redundancy resolution in R2 tool frame
class redundancy_resolution_dual(object):
    ###robot1 hold weld torch, positioner hold welded part
	def __init__(self,robot1,robot2,curve,curve_ori):
		self.robot1=robot1
		self.robot2=robot2
		self.curve=curve
		self.curve_ori=curve_ori

	def dual_arm_5dof_stepwise(self,q_init1,q_init2,w1=0.01,w2=0.01):
			###curve: x,y,z
			###curve_normal: i,j,k
			###q_init1: initial joint position for robot1 guess, no need to match perfectly
			###q_init2: initial joint position for robot2 guess, no need to match perfectly

			###w1: weight for first robot
			###w2: weight for second robot (larger weight path shorter)

			curve_normal=self.curve_ori

			###concatenated bounds
			upper_limit=np.hstack((self.robot1.upper_limit,self.robot2.upper_limit))
			lower_limit=np.hstack((self.robot1.lower_limit,self.robot2.lower_limit))

			q_out1=[q_init1]
			q_out2=[q_init2]


			#####weights
			Kw=0.1
			Kq=w1*np.eye(len(upper_limit))    #small value to make sure positive definite
			Kq[len(self.robot1.upper_limit):,len(self.robot1.upper_limit):]=w2*np.eye(len(self.robot2.upper_limit))		#larger weights for second robot for it moves slower
			lim_factor=1e-4

			q_cur1=copy.deepcopy(q_init1)
			q_cur2=copy.deepcopy(q_init2)

			for i in range(len(self.curve)):
				try:
					now=time.time()
					error_fb=999
					while error_fb>0.01:
						###timeout guard
						if time.time()-now>1:
							raise Exception("QP Timeout")

						pose1_world_now=self.robot1.fwd(q_cur1,world=True)
						pose2_now=self.robot2.fwd(q_cur2)
						pose2_world_now=self.robot2.fwd(q_cur2,world=True)

						p_cur = pose2_world_now.R.T@(pose1_world_now.p-pose2_world_now.p)
						Ep=self.curve[i]-p_cur
						R_cur=np.dot(pose2_world_now.R.T,pose1_world_now.R)

						error_fb=np.linalg.norm(Ep-self.curve[i])+np.linalg.norm(R_cur[:,-1]-curve_normal[i])	


						########################################################QP formation###########################################
						
						J1=self.robot1.jacobian(q_cur1)       #current Jacobian
						J1p=pose2_world_now.R.T@self.robot1.base_H[:3,:3]@J1[3:,:]		#convert to global frame then to robot2 tool frame
						J1R=pose2_world_now.R.T@self.robot1.base_H[:3,:3]@J1[:3,:]
						J1R_mod=-np.dot(hat(R_cur[:,-1]),J1R)

						J2=self.robot2.jacobian(q_cur2)        #calculate current Jacobian, mapped to robot2 tool frame
						J2p=np.dot(pose2_now.R.T,J2[3:,:])
						J2R=np.dot(pose2_now.R.T,J2[:3,:])
						J2R_mod=-np.dot(hat(R_cur[:,-1]),J2R)

						
						#form 6x12 jacobian with weight distribution, velocity propogate from rotation of TCP2
						J_all_p=np.hstack((J1p,-J2p+hat(p_cur)@J2R))
						J_all_R=np.hstack((J1R_mod,-J2R_mod))
						
						H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
						H=(H+np.transpose(H))/2


						ezdotd=curve_normal[i]-R_cur[:,-1]
						f=-np.dot(np.transpose(J_all_p),Ep)-Kw*np.dot(np.transpose(J_all_R),ezdotd)

						qdot=solve_qp(H,f,lb=lower_limit+lim_factor*np.ones(len(upper_limit))-np.hstack((q_cur1,q_cur2)),ub=upper_limit-lim_factor*np.ones(len(upper_limit))-np.hstack((q_cur1,q_cur2)),solver='quadprog')

						q_cur1+=qdot[:6]
						q_cur2+=qdot[6:]


				except:
					traceback.print_exc()
					raise Exception("QP failed")

				q_out1.append(q_cur1)
				q_out2.append(q_cur2)

			q_out1=np.array(q_out1)[1:]
			q_out2=np.array(q_out2)[1:]

			return q_out1, q_out2

	def feasibility_check(self,x):
		q_out1,q_out2 = self.dual_arm_6dof_stepwise(x[:len(self.robot1.upper_limit)],x[len(self.robot1.upper_limit):])
		return -len(q_out1)

	def dual_arm_6dof(self,q_init1,q_init2):
		###random quick search with optimization for a feasible solution

		upper_limit=np.hstack((self.robot1.upper_limit,self.robot2.upper_limit))
		lower_limit=np.hstack((self.robot1.lower_limit,self.robot2.lower_limit))
		bnds=tuple(zip(lower_limit,upper_limit))
		res = differential_evolution(self.feasibility_check, bnds, args=None,workers=-1,
										x0 = np.hstack((q_init1,q_init2)),
										strategy='best1bin', maxiter=1000,
										popsize=15, tol=1e-10,
										mutation=(0.5, 1), recombination=0.7,
										seed=None, callback=None, disp=True,
										polish=False, init='latinhypercube',
										atol=0)
		
		print(res)
		q_out1, q_out2=self.dual_arm_6dof_stepwise(res.x[:len(self.robot1.upper_limit)],res.x[len(self.robot1.upper_limit):])

		if len(q_out1)<len(self.curve):
			raise Exception("Infeasible")


		return q_out1, q_out2

	def dual_arm_6dof_stepwise(self,q_init1,q_init2,w1=0.01,w2=0.01):
			###curve: x,y,z
			###curve_R: Nx3x3 rotation matrix
			###q_init1: initial joint position for robot1 guess, no need to match perfectly
			###q_init2: initial joint position for robot2 guess, no need to match perfectly

			###w1: weight for first robot
			###w2: weight for second robot (larger weight path shorter)

			curve_R=self.curve_ori
			###concatenated bounds
			upper_limit=np.hstack((self.robot1.upper_limit,self.robot2.upper_limit))
			lower_limit=np.hstack((self.robot1.lower_limit,self.robot2.lower_limit))

			q_out1=[q_init1]
			q_out2=[q_init2]


			#####weights
			Kw=0.1
			Kq=w1*np.eye(len(upper_limit))    #small value to make sure positive definite
			Kq[len(self.robot1.upper_limit):,len(self.robot1.upper_limit):]=w2*np.eye(len(self.robot2.upper_limit))		#larger weights for second robot for it moves slower
			lim_factor=1e-4

			q_cur1=copy.deepcopy(q_init1)
			q_cur2=copy.deepcopy(q_init2)

			for i in range(len(self.curve)):
				# print(i)
				try:
					now=time.time()
					error_fb=999
					while error_fb>0.001:
						###timeout guard
						if time.time()-now>1:
							raise Exception("QP Timeout")

						pose1_world_now=self.robot1.fwd(q_cur1,world=True)
						pose2_now=self.robot2.fwd(q_cur2)
						pose2_world_now=self.robot2.fwd(q_cur2,world=True)

						p_cur = pose2_world_now.R.T@(pose1_world_now.p-pose2_world_now.p)
						Ep=self.curve[i]-p_cur
						R_cur=np.dot(pose2_world_now.R.T,pose1_world_now.R)
						ER=curve_R[i]@R_cur.T
						
						# print(np.linalg.norm(Ep),np.linalg.norm(ER-np.eye(3)))
						error_fb=np.linalg.norm(Ep)+np.linalg.norm(ER-np.eye(3))	


						########################################################QP formation###########################################
						
						J1=self.robot1.jacobian(q_cur1)       					#current Jacobian
						J1p=pose2_world_now.R.T@self.robot1.base_H[:3,:3]@J1[3:,:]		#convert to global frame then to robot2 tool frame
						J1R=pose2_world_now.R.T@self.robot1.base_H[:3,:3]@J1[:3,:]

						J2=self.robot2.jacobian(q_cur2)        #calculate current Jacobian, mapped to robot2 tool frame
						J2p=np.dot(pose2_now.R.T,J2[3:,:])
						J2R=np.dot(pose2_now.R.T,J2[:3,:])

						
						#form 6x12 jacobian with weight distribution, velocity propogate from rotation of TCP2
						J_all_p=np.hstack((J1p,-J2p+hat(p_cur)@J2R))
						J_all_R=np.hstack((J1R,-J2R))
						
						H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
						H=(H+np.transpose(H))/2

						k,theta = R2rot(ER)
						wd=np.sin(theta/2)*k 
						

						f=-np.dot(np.transpose(J_all_p),Ep)-Kw*np.dot(np.transpose(J_all_R),wd)

						qdot=solve_qp(H,f,lb=lower_limit+lim_factor*np.ones(len(upper_limit))-np.hstack((q_cur1,q_cur2)),ub=upper_limit-lim_factor*np.ones(len(upper_limit))-np.hstack((q_cur1,q_cur2)),solver='quadprog')

						q_cur1+=qdot[:6]
						q_cur2+=qdot[6:]


				except:
					traceback.print_exc()
					break

				q_out1.append(copy.deepcopy(q_cur1))
				q_out2.append(copy.deepcopy(q_cur2))

			q_out1=np.array(q_out1)[1:]
			q_out2=np.array(q_out2)[1:]

			return q_out1, q_out2