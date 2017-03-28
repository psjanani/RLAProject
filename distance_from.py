import numpy as np

ACTION_DELTAS = [
	[-1, 0],
	[0, 1],
	[1, 0],
	[0, -1]
]

NUM_A = len(ACTION_DELTAS)

def smart_move(barrier, my_pts, other_pts, direction='closer'):
	distance = distanceFrom(barrier, other_pts)
	ROWS, COLS = np.shape(barrier)

	n = np.shape(my_pts)[0]

	actions = []

	for pt in my_pts:
		distances = np.zeros([NUM_A, 1])

		for j in range(NUM_A):
			new_pt = (pt[0] + ACTION_DELTAS[j][0],  pt[1] + ACTION_DELTAS[j][1])

			if new_pt[0] < 0 or new_pt[1] < 0 or new_pt[0] >= ROWS or new_pt[1] >= COLS:
				distances[j] = distance[pt]
			else:
				distances[j] = distance[new_pt]

		if direction == 'closer':
			actions.append(int(np.argmin(distances)))
		else:
			actions.append(int(np.argmax(distances)))

	return actions


def distanceFrom(barrier, pts):
	ROWS, COLS = np.shape(barrier)

	distance = np.zeros([ROWS, COLS])
	distance[:, :] = -1

	q = []

	for pt in pts:
		q.append(pt)
		distance[pt] = 0

	steps = 0
	while len(q) > 0:
		next_q = []
		steps += 1

		while len(q) > 0:
			r,c = q.pop()

			if r - 1 >= 0 and distance[r - 1, c] < 0 and barrier[r - 1][c] == 0:
				distance[r - 1, c] = steps
				next_q.append((r - 1, c))

			if c - 1 >= 0 and distance[r, c - 1] < 0 and barrier[r][c - 1] == 0:
				distance[r, c - 1] = steps
				next_q.append((r, c - 1))

			if r + 1 < ROWS and distance[r + 1, c] < 0 and barrier[r + 1][c] == 0:
				distance[r + 1, c] = steps
				next_q.append((r + 1, c))

			if c + 1 < COLS and distance[r, c + 1] < 0 and barrier[r][c + 1] == 0:
				distance[r, c + 1] = steps
				next_q.append((r, c + 1))

		q = next_q

	return distance

if __name__ == "__main__":
	print "for demo_purposes..."
	barrier = np.zeros([10, 10])
	barrier[1:4, 2:] = 1
	barrier[5:8, 0:8] = 1
	distance = distanceFrom(barrier, [(0,0), (1, 1)])

	for r in range(10):
		for c in range(10):
			if barrier[r, c] == 1:
				print "_\t",
			elif (r,c) in pts:
				print "*\t", 
			else:
				print str(distance[r,c]) + "\t",

	print ""