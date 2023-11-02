y = [0,0,0,1,0,0,1,0,0,1,1,0,1,1,1,1]
x = [0.1,0.1,0.25,0.25,0.3,0.33,0.4,0.52,0.55,0.7,0.8,0.85,0.9,0.9,0.95,1]

for t in [0, 0.2, 0.4, 0.6, 0.8, 1]:
	TruePositives = 0
	FalsePositives = 0
	FalseNegatives = 0
	for i in range(len(x)):
		if (y[i]):
			if (x[i] > t):
				TruePositives += 1
			else:
				FalseNegatives += 1
		elif (not y[i] and x[i] > t):
			FalsePositives += 1
		
	if (TruePositives + FalsePositives == 0):
		Recall = -1
	else:
		Recall = TruePositives / (TruePositives + FalsePositives)
		
	if (TruePositives + FalseNegatives == 0):
		Precision = -1
	else:
		Precision = TruePositives / (TruePositives + FalseNegatives)
	print("t = {:.1f}\tRecall = {:.4f}\tPrecision = {:.4f}".format(t, Recall, Precision))
	print("TruePositives = {:d}\tFalsePositives = {:d}\tFalseNegatives = {:d}".format(TruePositives,FalsePositives,FalseNegatives))
	print()
