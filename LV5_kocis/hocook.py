import numpy as np

def interpolate(Q, DQ, t):
	n = Q.shape[0]
	m = Q.shape[1]
	dt = t[1:] - t[:-1]

	p0 = np.zeros((n,m-1))
	p1 = np.zeros((n,m-1))
	p2 = np.zeros((n,m-1))
	p3 = np.zeros((n,m-1))
	p4 = np.zeros((n,m-1))
	# p0[:,0] = Q[:,0]
	# p3[:,0] = 4/dt[1]**3*(-Q[:,0]+Q[:,1])-1/dt[1]**2*DQ[:,1]
	# p4[:,0] = 3/dt[1]**4*(Q[:,0]-Q[:,1])+1/dt[1]**3*DQ[:,1]
	# for i in range(1,m-2):
	for i in range(m-1):
		p0[:,i] = Q[:,i]
		p1[:,i] = DQ[:,i]
		p2[:,i] = 3*(Q[:,i+1]-Q[:,i])/(t[i+1]-t[i])**2-(2*DQ[:,i]+DQ[:,i+1])/(t[i+1]-t[i])
		p3[:,i] = -2*(Q[:,i+1]-Q[:,i])/(t[i+1]-t[i])**3+(DQ[:,i]+DQ[:,i+1])/(t[i+1]-t[i])**2
		p4[:,i] = np.zeros(n)
	# p0[:,m-2] = Q[:,m-2]
	# p1[:,m-2] = DQ[:,m-2]
	# p2[:,m-2] = 6/(t[m-1]-t[m-2])**2*(-Q[:,m-2]+Q[:,m-1])-3/(t[m-1]-t[m-2])*DQ[:,m-2]
	# p3[:,m-2] = 8/(t[m-1]-t[m-2])**3*(Q[:,m-2]-Q[:,m-1]) +3/(t[m-1]-t[m-2])**2*DQ[:,m-2]
	# p4[:,m-2] = 3/(t[m-1]-t[m-2])**4*(-Q[:,m-2]+Q[:,m-1])-1/(t[m-1]-t[m-2])**3*DQ[:,m-2]

	A = np.stack((p0, p1, p2, p3, p4))

	return A

def hocook(Q, dqgr, ddqgr, Ts):
	n = Q.shape[0]
	m = Q.shape[1]

	dQ = Q[:,1:m] - Q[:,0:m-1]

	dt = np.concatenate((np.array([0]), np.sqrt((dQ**2).sum(0))))

	S = 0

	while S < 0.99 or S > 1.01:
		t = np.cumsum(dt)

		M = np.zeros((m-2,m-2))
		M[0,0] = 3/dt[1]+2/dt[2]
		M[1,0] = 1/dt[2]   
		for i in range(1,m-3):
			M[i-1,i] = dt[i+2]
			M[i,i] = 2*(dt[i+1]+dt[i+2])
			M[i+1,i] = dt[i+1]
		M[m-4,m-3] = 1/dt[m-2]
		M[m-3,m-3] = 2/dt[m-2]+3/dt[m-1]
		
		A = np.zeros((n,m-2))
		A[:,0] = 6/dt[1]**2*(Q[:,1]-Q[:,0])+3/dt[2]**2*(Q[:,2]-Q[:,1])
		for i in range(1,m-3):
			A[:,i] = 3/(dt[i+1]*dt[i+2])*(dt[i+1]**2*(Q[:,i+2]-Q[:,i+1])+dt[i+2]**2*(Q[:,i+1]-Q[:,i]))
		A[:,m-3] = 3/dt[m-2]**2*(Q[:,m-2]-Q[:,m-3])+6/dt[m-1]**2*(Q[:,m-1]-Q[:,m-2])

		DQ = np.concatenate((np.zeros((n,1)), A @ np.linalg.inv(M), np.zeros((n,1))), 1)

		P = interpolate(Q, DQ, t)

		p0 = P[0,:,:]
		p1 = P[1,:,:]
		p2 = P[2,:,:]
		p3 = P[3,:,:]
		p4 = P[4,:,:]

		# p0 = np.zeros((n,m-1))
		# p1 = np.zeros((n,m-1))
		# p2 = np.zeros((n,m-1))
		# p3 = np.zeros((n,m-1))
		# p4 = np.zeros((n,m-1))
		# p0[:,0] = Q[:,0]
		# p3[:,0] = 4/dt[1]**3*(-Q[:,0]+Q[:,1])-1/dt[1]**2*DQ[:,1]
		# p4[:,0] = 3/dt[1]**4*(Q[:,0]-Q[:,1])+1/dt[1]**3*DQ[:,1]
		# for i in range(1,m-2):
		# 	p0[:,i] = Q[:,i]
		# 	p1[:,i] = DQ[:,i]
		# 	p2[:,i] = 3*(Q[:,i+1]-Q[:,i])/(t[i+1]-t[i])**2-(2*DQ[:,i]+DQ[:,i+1])/(t[i+1]-t[i])
		# 	p3[:,i] = -2*(Q[:,i+1]-Q[:,i])/(t[i+1]-t[i])**3+(DQ[:,i]+DQ[:,i+1])/(t[i+1]-t[i])**2
		# 	p4[:,i] = np.zeros(n)
		# p0[:,m-2] = Q[:,m-2]
		# p1[:,m-2] = DQ[:,m-2]
		# p2[:,m-2] = 6/(t[m-1]-t[m-2])**2*(-Q[:,m-2]+Q[:,m-1])-3/(t[m-1]-t[m-2])*DQ[:,m-2]
		# p3[:,m-2] = 8/(t[m-1]-t[m-2])**3*(Q[:,m-2]-Q[:,m-1]) +3/(t[m-1]-t[m-2])**2*DQ[:,m-2]
		# p4[:,m-2] = 3/(t[m-1]-t[m-2])**4*(-Q[:,m-2]+Q[:,m-1])-1/(t[m-1]-t[m-2])**3*DQ[:,m-2]
		
		nk = np.floor(t[m-1] / Ts).astype('int') + 1
		Qc = np.zeros((n,nk))
		dQc = np.zeros((n,nk))
		ddQc = np.zeros((n,nk))
		tc = Ts * np.arange(nk)
		for k in range(nk):
			i = np.argmax(t > tc[k]) - 1
			tcr = tc[k] - t[i]
			Qc[:,k] = p0[:,i]+p1[:,i]*tcr+p2[:,i]*tcr**2+p3[:,i]*tcr**3+p4[:,i]*tcr**4
			dQc[:,k] = p1[:,i]+2*p2[:,i]*tcr+3*p3[:,i]*tcr**2+4*p4[:,i]*tcr**3
			ddQc[:,k] = 2*p2[:,i]+6*p3[:,i]*tcr+12*p4[:,i]*tcr**2
			
		maxddq = np.abs(2*p2)
		quartic = (p4 != 0)
		text = -np.ones((n,m-1))
		text[quartic] = -p3[quartic] / (4*p4[quartic])
		extrem = np.logical_and(text > 0, text <= dt[1:])
		maxddq[extrem] = np.maximum(maxddq[extrem], np.abs(2*p2[extrem]+6*p3[extrem]*text[extrem]+12*p4[extrem]*text[extrem]**2))
		maxddq = maxddq.max(1)

		maxdq = np.abs(DQ).max(1)
		for i in range(m-1):
			for j in range(n):
				text = np.roots([12*p4[j,i], 6*p3[j,i], 2*p2[j,i]])
				ntext = text.shape[0]
				for k in range(ntext):
					if text[k] > 0 and text[k] <= dt[i+1]:
						dq = np.abs(p1[j,i]+2*p2[j,i]*text[k]+3*p3[j,i]*text[k]**2+4*p4[j,i]*text[k]**3)						
						if dq > maxdq[j]:
							maxdq[j] = dq
		
		S = max((maxddq/ddqgr).max(), (maxdq/dqgr).max())

		print('S=%f' % S)

		dt *= np.sqrt(S)

	return Qc, dQc, ddQc, tc