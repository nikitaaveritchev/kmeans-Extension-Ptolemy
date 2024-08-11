import numpy as np

class DataGenerator:
    def __init__(self, n_clusters=3, n_samples=300, n_features=2, random_state=42):
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def generate_data(self):
        np.random.seed(0)
        means = np.random.uniform(0, 10, (self.n_clusters, self.n_features))
        covariances = np.array([np.eye(self.n_features) for _ in range(self.n_clusters)])
        n_samples_per_cluster = self.n_samples // self.n_clusters
        data = np.zeros((self.n_samples, self.n_features))
        labels = np.zeros(self.n_samples, dtype=int)

        for i in range(self.n_clusters):
            start = i * n_samples_per_cluster
            end = start + n_samples_per_cluster
            data[start:end, :] = np.random.multivariate_normal(means[i], covariances[i], n_samples_per_cluster)
            labels[start:end] = i

        remaining_samples = self.n_samples % self.n_clusters
        if remaining_samples > 0:
            for i in range(remaining_samples):
                data[-(i+1), :] = np.random.multivariate_normal(means[i], covariances[i], 1)
                labels[-(i+1)] = i

        permutation = np.random.permutation(self.n_samples)
        data = data[permutation]
        labels = labels[permutation]

        return data, labels

class Kmeans:
    def __init__(self, k=2, max_iter=250, tolerance=0.000001, method='Elkan'):
        assert method in ['classic', 'Elkan', 'Ptolemy_upper', 'Ptolemy_lower', 'Ptolemy'], "Method argument not valid"
        self.k = k
        self.max_iter = max_iter
        self.tol = tolerance
        self.method = method
        self.distance_evaluations = 0

    def fit(self, data):
        pointsArray = np.array(data)
        self.centroids = {}
        self.labels = [0 for point in pointsArray]
        initCentroids = []
        np.random.seed(42)

        for dim in range(pointsArray.shape[1]):
            dim_min = np.min(pointsArray, axis=0)[dim]
            dim_max = np.max(pointsArray, axis=0)[dim]
            newCentroid = (dim_max - dim_min) * np.random.random_sample([1, self.k]) + dim_min
            initCentroids = np.append(initCentroids, newCentroid)

        initCentroids = initCentroids.reshape((pointsArray.shape[1], self.k)).T
        self.centroids = dict(zip(list(range(self.k)), initCentroids))

        if self.method == 'classic':
            self._classic_kmeans(pointsArray)
        elif self.method == 'Elkan':
            self._elkan_kmeans(pointsArray)
        elif self.method == 'Ptolemy_upper':
            self._ptolemy_upper_kmeans(pointsArray)
        elif self.method == 'Ptolemy_lower':
            self._ptolemy_lower_kmeans(pointsArray)
        elif self.method == 'Ptolemy':
            self._ptolemy_kmeans(pointsArray)

    def _classic_kmeans(self, pointsArray):
        for i in range(self.max_iter):
            self.classifications = {i: [] for i in range(self.k)}
            self.pointsClassif = {i: [] for i in range(self.k)}

            for i, point in enumerate(pointsArray):
                distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
                self.distance_evaluations += self.k
                classification = distances.index(min(distances))
                self.classifications[classification].append(point)
                self.pointsClassif[classification].append(i)

            prevCentroids = dict(self.centroids)

            for classification in self.classifications:
                if len(self.classifications[classification]) > 0:
                    self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            for centroid in self.centroids:
                original_centroid = prevCentroids[centroid]
                current_centroid = self.centroids[centroid]
                if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                    optimized = False
            if optimized:
                for centroid in self.pointsClassif:
                    for point in self.pointsClassif[centroid]:
                        self.labels[point] = centroid
                break

    def _elkan_kmeans(self, pointsArray):
        lowerBounds = np.zeros((pointsArray.shape[0], self.k))
        upperBounds = np.zeros(pointsArray.shape[0])
        self.classifications = {i: [] for i in range(self.k)}
        self.pointsClassif = {i: [] for i in range(self.k)}

        for i, point in enumerate(pointsArray):
            distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
            self.distance_evaluations += self.k
            classification = distances.index(min(distances))
            self.classifications[classification].append(point)
            self.pointsClassif[classification].append(i)
            lowerBounds[i] = distances
            upperBounds[i] = min(distances)

        prevCentroids = dict(self.centroids)
        prevPointsClassif = dict(self.pointsClassif)
        
        for classification in self.classifications:
            if len(self.classifications[classification]) > 0:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

        optimized = True
        centroidDistanceChange = {}

        for centroid in self.centroids:
            original_centroid = prevCentroids[centroid]
            current_centroid = self.centroids[centroid]
            centroidDistanceChange[centroid] = np.linalg.norm(original_centroid - current_centroid)
            if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                optimized = False

        if optimized:
            for centroid in self.pointsClassif:
                for point in self.pointsClassif[centroid]:
                    self.labels[point] = centroid
            return

        for centroid in self.pointsClassif:
            for i in range(pointsArray.shape[0]):
                lowerBounds[i][centroid] -= centroidDistanceChange[centroid]
            for i in self.pointsClassif[centroid]:
                upperBounds[i] += centroidDistanceChange[centroid]

        for it in range(self.max_iter):
            listCentroids = list(range(self.k))
            self.classifications = {i: [] for i in range(self.k)}
            self.pointsClassif = {i: [] for i in range(self.k)}
            centroidDistances = {}
            closestCentroidDistances = {}

            for i in range(self.k):
                centroidDistances[i] = [np.linalg.norm(self.centroids[i] - self.centroids[c_prime]) for c_prime in self.centroids]
                closestCentroidDistances[i] = min(centroidDistances[i][:i] + centroidDistances[i][i + 1:])

            for centroid in prevPointsClassif:
                for i in prevPointsClassif[centroid]:
                    r = True
                    distToCurrentCentroid = upperBounds[i]

                    if upperBounds[i] <= 0.5 * closestCentroidDistances[centroid]:
                        self.classifications[centroid].append(pointsArray[i])
                        self.pointsClassif[centroid].append(i)
                    else:
                        assigned_centroid = centroid
                        for c_prime in (listCentroids[:centroid] + listCentroids[centroid + 1:]):
                            if (distToCurrentCentroid > lowerBounds[i][c_prime]) and (distToCurrentCentroid > 0.5 * centroidDistances[centroid][c_prime]):
                                if r:
                                    distToCurrentCentroid = np.linalg.norm(pointsArray[i] - self.centroids[centroid])
                                    self.distance_evaluations += 1
                                    r = False
                                distToCPrime = np.linalg.norm(pointsArray[i] - self.centroids[c_prime])
                                self.distance_evaluations += 1  # Count distance computation
                                if distToCurrentCentroid > distToCPrime:
                                    assigned_centroid = c_prime
                        self.classifications[assigned_centroid].append(pointsArray[i])
                        self.pointsClassif[assigned_centroid].append(i)

            prevCentroids = dict(self.centroids)
            prevPointsClassif = dict(self.pointsClassif)
            
            for classification in self.classifications:
                if len(self.classifications[classification]) > 0:
                    self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            centroidDistanceChange = {}

            for centroid in self.centroids:
                original_centroid = prevCentroids[centroid]
                current_centroid = self.centroids[centroid]
                centroidDistanceChange[centroid] = np.linalg.norm(original_centroid - current_centroid)
                if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                    optimized = False
            if optimized:
                break

            for centroid in self.pointsClassif:
                for i in range(pointsArray.shape[0]):
                    lowerBounds[i][centroid] -= centroidDistanceChange[centroid]
                for i in self.pointsClassif[centroid]:
                    upperBounds[i] += centroidDistanceChange[centroid]

        for centroid in self.pointsClassif:
            for point in self.pointsClassif[centroid]:
                self.labels[point] = centroid


    def _ptolemy_upper_kmeans(self, pointsArray):
        lowerBounds = np.zeros((pointsArray.shape[0], self.k))
        upperBounds = np.zeros(pointsArray.shape[0])
        self.classifications = {i: [] for i in range(self.k)}
        self.pointsClassif = {i: [] for i in range(self.k)}

        for i, point in enumerate(pointsArray):
            distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
            self.distance_evaluations += self.k
            classification = distances.index(min(distances))
            self.classifications[classification].append(point)
            self.pointsClassif[classification].append(i)
            lowerBounds[i] = distances
            upperBounds[i] = min(distances)

        prevCentroids = dict(self.centroids)
        prevPointsClassif = dict(self.pointsClassif)

        for classification in self.classifications:
            if len(self.classifications[classification]) > 0:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

        optimized = True
        centroidDistanceChange = {}

        for centroid in self.centroids:
            original_centroid = prevCentroids[centroid]
            current_centroid = self.centroids[centroid]
            centroidDistanceChange[centroid] = np.linalg.norm(original_centroid - current_centroid)
            if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                optimized = False
        if optimized:
            for centroid in self.pointsClassif:
                for point in self.pointsClassif[centroid]:
                    self.labels[point] = centroid
            return

        oldLowerBounds = lowerBounds.copy()
        oldUpperBounds = upperBounds.copy()
        prevDistanceChange = centroidDistanceChange.copy()
        oldPrevCentroids = prevCentroids.copy()

        for centroid in self.pointsClassif:
            for i in range(pointsArray.shape[0]):
                lowerBounds[i][centroid] -= centroidDistanceChange[centroid]
            for i in self.pointsClassif[centroid]:
                upperBounds[i] += centroidDistanceChange[centroid]

        for it in range(self.max_iter):
            listCentroids = list(range(self.k))
            self.classifications = {i: [] for i in range(self.k)}
            self.pointsClassif = {i: [] for i in range(self.k)}
            centroidDistances = {}
            closestCentroidDistances = {}

            for i in range(self.k):
                centroidDistances[i] = [np.linalg.norm(self.centroids[i] - self.centroids[c_prime]) for c_prime in self.centroids]
                closestCentroidDistances[i] = min(centroidDistances[i][:i] + centroidDistances[i][i + 1:])

            for centroid in prevPointsClassif:
                for i in prevPointsClassif[centroid]:
                    r = True
                    distToCurrentCentroid = upperBounds[i]

                    if upperBounds[i] <= 0.5 * closestCentroidDistances[centroid]:
                        self.classifications[centroid].append(pointsArray[i])
                        self.pointsClassif[centroid].append(i)
                    else:
                        assigned_centroid = centroid
                        for c_prime in (listCentroids[:centroid] + listCentroids[centroid + 1:]):
                            if (distToCurrentCentroid > lowerBounds[i][c_prime]) and (distToCurrentCentroid > 0.5 * centroidDistances[centroid][c_prime]):
                                if r:
                                    distToCurrentCentroid = np.linalg.norm(pointsArray[i] - self.centroids[centroid])
                                    self.distance_evaluations += 1
                                    r = False
                                distToCPrime = np.linalg.norm(pointsArray[i] - self.centroids[c_prime])
                                self.distance_evaluations += 1  # Count distance computation
                                if distToCurrentCentroid > distToCPrime:
                                    assigned_centroid = c_prime
                        self.classifications[assigned_centroid].append(pointsArray[i])
                        self.pointsClassif[assigned_centroid].append(i)

            prevCentroids = dict(self.centroids)
            prevPointsClassif = dict(self.pointsClassif)

            for classification in self.classifications:
                if len(self.classifications[classification]) > 0:
                    self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            centroidDistanceChange = {}

            for centroid in self.centroids:
                original_centroid = prevCentroids[centroid]
                current_centroid = self.centroids[centroid]
                centroidDistanceChange[centroid] = np.linalg.norm(original_centroid - current_centroid)
                if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                    optimized = False
            if optimized:
                break
            
            # Distance old previous centroids to new centroids
            oldCentroidDistanceChange = {}
            for centroid in self.centroids:
                old_centroid = oldPrevCentroids[centroid]
                current_centroid = self.centroids[centroid]
                oldCentroidDistanceChange[centroid] = np.linalg.norm(old_centroid - current_centroid)

            for centroid in self.pointsClassif:
                for i in range(pointsArray.shape[0]):
                    lowerBounds[i][centroid] = max(0, lowerBounds[i][centroid] - centroidDistanceChange[centroid])
                for i in self.pointsClassif[centroid]:
                    if prevDistanceChange[centroid] != 0:

                        upperBounds[i] = (upperBounds[i] * oldCentroidDistanceChange[centroid] + oldUpperBounds[i] * centroidDistanceChange[centroid]) / prevDistanceChange[centroid]
                    else:
                        upperBounds[i] += centroidDistanceChange[centroid]
                    upperBounds[i] = max(upperBounds[i], lowerBounds[i][centroid])

            oldLowerBounds = lowerBounds.copy()
            oldUpperBounds = upperBounds.copy()
            prevDistanceChange = centroidDistanceChange.copy()
            oldPrevCentroids = self.centroids.copy()

        for centroid in self.pointsClassif:
            for point in self.pointsClassif[centroid]:
                self.labels[point] = centroid

    def _ptolemy_lower_kmeans(self, pointsArray):
        lowerBounds = np.zeros((pointsArray.shape[0], self.k))
        upperBounds = np.zeros(pointsArray.shape[0])
        self.classifications = {i: [] for i in range(self.k)}
        self.pointsClassif = {i: [] for i in range(self.k)}

        for i, point in enumerate(pointsArray):
            distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
            self.distance_evaluations += self.k
            classification = distances.index(min(distances))
            self.classifications[classification].append(point)
            self.pointsClassif[classification].append(i)
            lowerBounds[i] = distances
            upperBounds[i] = min(distances)

        prevCentroids = dict(self.centroids)
        prevPointsClassif = dict(self.pointsClassif)

        for classification in self.classifications:
            if len(self.classifications[classification]) > 0:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

        optimized = True
        centroidDistanceChange = {}

        for centroid in self.centroids:
            original_centroid = prevCentroids[centroid]
            current_centroid = self.centroids[centroid]
            centroidDistanceChange[centroid] = np.linalg.norm(original_centroid - current_centroid)
            if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                optimized = False
        if optimized:
            for centroid in self.pointsClassif:
                for point in self.pointsClassif[centroid]:
                    self.labels[point] = centroid
            return

        oldLowerBounds = lowerBounds.copy()
        oldUpperBounds = upperBounds.copy()
        prevDistanceChange = centroidDistanceChange.copy()
        oldPrevCentroids = prevCentroids.copy()

        for centroid in self.pointsClassif:
            for i in range(pointsArray.shape[0]):
                lowerBounds[i][centroid] -= centroidDistanceChange[centroid]
            for i in self.pointsClassif[centroid]:
                upperBounds[i] += centroidDistanceChange[centroid]

        for it in range(self.max_iter):
            listCentroids = list(range(self.k))
            self.classifications = {i: [] for i in range(self.k)}
            self.pointsClassif = {i: [] for i in range(self.k)}
            centroidDistances = {}
            closestCentroidDistances = {}

            for i in range(self.k):
                centroidDistances[i] = [np.linalg.norm(self.centroids[i] - self.centroids[c_prime]) for c_prime in self.centroids]
                closestCentroidDistances[i] = min(centroidDistances[i][:i] + centroidDistances[i][i + 1:])

            for centroid in prevPointsClassif:
                for i in prevPointsClassif[centroid]:
                    r = True
                    distToCurrentCentroid = upperBounds[i]

                    if upperBounds[i] <= 0.5 * closestCentroidDistances[centroid]:
                        self.classifications[centroid].append(pointsArray[i])
                        self.pointsClassif[centroid].append(i)
                    else:
                        assigned_centroid = centroid
                        for c_prime in (listCentroids[:centroid] + listCentroids[centroid + 1:]):
                            if (distToCurrentCentroid > lowerBounds[i][c_prime]) and (distToCurrentCentroid > 0.5 * centroidDistances[centroid][c_prime]):
                                if r:
                                    distToCurrentCentroid = np.linalg.norm(pointsArray[i] - self.centroids[centroid])
                                    self.distance_evaluations += 1
                                    r = False
                                distToCPrime = np.linalg.norm(pointsArray[i] - self.centroids[c_prime])
                                self.distance_evaluations += 1  # Count distance computation
                                if distToCurrentCentroid > distToCPrime:
                                    assigned_centroid = c_prime
                        self.classifications[assigned_centroid].append(pointsArray[i])
                        self.pointsClassif[assigned_centroid].append(i)

            prevCentroids = dict(self.centroids)
            prevPointsClassif = dict(self.pointsClassif)

            for classification in self.classifications:
                if len(self.classifications[classification]) > 0:
                    self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            centroidDistanceChange = {}

            for centroid in self.centroids:
                original_centroid = prevCentroids[centroid]
                current_centroid = self.centroids[centroid]
                centroidDistanceChange[centroid] = np.linalg.norm(original_centroid - current_centroid)
                if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                    optimized = False
            if optimized:
                break
            
            # Distance old previous centroids to new centroids
            oldCentroidDistanceChange = {}
            for centroid in self.centroids:
                old_centroid = oldPrevCentroids[centroid]
                current_centroid = self.centroids[centroid]
                oldCentroidDistanceChange[centroid] = np.linalg.norm(old_centroid - current_centroid)

            for centroid in self.pointsClassif:
                for i in range(pointsArray.shape[0]):
                    if prevDistanceChange[centroid] != 0:
                        lowerbound1 = (oldLowerBounds[i][centroid] * centroidDistanceChange[centroid] - upperBounds[i]*oldCentroidDistanceChange[centroid])/ prevDistanceChange[centroid]
                        lowerbound2 = (lowerBounds[i][centroid] * oldCentroidDistanceChange[centroid] - oldUpperBounds[i]*centroidDistanceChange[centroid])/ prevDistanceChange[centroid]
                        if lowerbound1 < 0 and lowerbound2 < 0:
                            lowerBounds[i][centroid] = 0
                        else:
                            lowerBounds[i][centroid] = max(lowerbound1, lowerbound2)
                    else:
                        lowerBounds[i][centroid] = max(0, lowerBounds[i][centroid] - centroidDistanceChange[centroid])
                for i in self.pointsClassif[centroid]:
                    upperBounds[i] += centroidDistanceChange[centroid]


            oldLowerBounds = lowerBounds.copy()
            oldUpperBounds = upperBounds.copy()
            prevDistanceChange = centroidDistanceChange.copy()
            oldPrevCentroids = self.centroids.copy()

        for centroid in self.pointsClassif:
            for point in self.pointsClassif[centroid]:
                self.labels[point] = centroid

    def _ptolemy_kmeans(self, pointsArray):
        lowerBounds = np.zeros((pointsArray.shape[0], self.k))
        upperBounds = np.zeros(pointsArray.shape[0])
        self.classifications = {i: [] for i in range(self.k)}
        self.pointsClassif = {i: [] for i in range(self.k)}

        for i, point in enumerate(pointsArray):
            distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
            self.distance_evaluations += self.k
            classification = distances.index(min(distances))
            self.classifications[classification].append(point)
            self.pointsClassif[classification].append(i)
            lowerBounds[i] = distances
            upperBounds[i] = min(distances)

        prevCentroids = dict(self.centroids)
        prevPointsClassif = dict(self.pointsClassif)

        for classification in self.classifications:
            if len(self.classifications[classification]) > 0:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

        optimized = True
        centroidDistanceChange = {}

        for centroid in self.centroids:
            original_centroid = prevCentroids[centroid]
            current_centroid = self.centroids[centroid]
            centroidDistanceChange[centroid] = np.linalg.norm(original_centroid - current_centroid)
            if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                optimized = False
        if optimized:
            for centroid in self.pointsClassif:
                for point in self.pointsClassif[centroid]:
                    self.labels[point] = centroid
            return

        oldLowerBounds = lowerBounds.copy()
        oldUpperBounds = upperBounds.copy()
        prevDistanceChange = centroidDistanceChange.copy()
        oldPrevCentroids = prevCentroids.copy()

        for centroid in self.pointsClassif:
            for i in range(pointsArray.shape[0]):
                lowerBounds[i][centroid] -= centroidDistanceChange[centroid]
            for i in self.pointsClassif[centroid]:
                upperBounds[i] += centroidDistanceChange[centroid]

        for it in range(self.max_iter):
            listCentroids = list(range(self.k))
            self.classifications = {i: [] for i in range(self.k)}
            self.pointsClassif = {i: [] for i in range(self.k)}
            centroidDistances = {}
            closestCentroidDistances = {}

            for i in range(self.k):
                centroidDistances[i] = [np.linalg.norm(self.centroids[i] - self.centroids[c_prime]) for c_prime in self.centroids]
                closestCentroidDistances[i] = min(centroidDistances[i][:i] + centroidDistances[i][i + 1:])

            for centroid in prevPointsClassif:
                for i in prevPointsClassif[centroid]:
                    r = True
                    distToCurrentCentroid = upperBounds[i]

                    if upperBounds[i] <= 0.5 * closestCentroidDistances[centroid]:
                        self.classifications[centroid].append(pointsArray[i])
                        self.pointsClassif[centroid].append(i)
                    else:
                        assigned_centroid = centroid
                        for c_prime in (listCentroids[:centroid] + listCentroids[centroid + 1:]):
                            if (distToCurrentCentroid > lowerBounds[i][c_prime]) and (distToCurrentCentroid > 0.5 * centroidDistances[centroid][c_prime]):
                                if r:
                                    distToCurrentCentroid = np.linalg.norm(pointsArray[i] - self.centroids[centroid])
                                    self.distance_evaluations += 1
                                    r = False
                                distToCPrime = np.linalg.norm(pointsArray[i] - self.centroids[c_prime])
                                self.distance_evaluations += 1  # Count distance computation
                                if distToCurrentCentroid > distToCPrime:
                                    assigned_centroid = c_prime
                        self.classifications[assigned_centroid].append(pointsArray[i])
                        self.pointsClassif[assigned_centroid].append(i)

            prevCentroids = dict(self.centroids)
            prevPointsClassif = dict(self.pointsClassif)

            for classification in self.classifications:
                if len(self.classifications[classification]) > 0:
                    self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            centroidDistanceChange = {}

            for centroid in self.centroids:
                original_centroid = prevCentroids[centroid]
                current_centroid = self.centroids[centroid]
                centroidDistanceChange[centroid] = np.linalg.norm(original_centroid - current_centroid)
                if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                    optimized = False
            if optimized:
                break
            
            # Distance old previous centroids to new centroids
            oldCentroidDistanceChange = {}
            for centroid in self.centroids:
                old_centroid = oldPrevCentroids[centroid]
                current_centroid = self.centroids[centroid]
                oldCentroidDistanceChange[centroid] = np.linalg.norm(old_centroid - current_centroid)

            for centroid in self.pointsClassif:
                for i in range(pointsArray.shape[0]):
                    if prevDistanceChange[centroid] != 0:
                        lowerbound1 = (oldLowerBounds[i][centroid] * centroidDistanceChange[centroid] - upperBounds[i]*oldCentroidDistanceChange[centroid])/ prevDistanceChange[centroid]
                        lowerbound2 = (lowerBounds[i][centroid] * oldCentroidDistanceChange[centroid] - oldUpperBounds[i]*centroidDistanceChange[centroid])/ prevDistanceChange[centroid]
                        if lowerbound1 < 0 and lowerbound2 < 0:
                            lowerBounds[i][centroid] = 0
                        else:
                            lowerBounds[i][centroid] = max(lowerbound1, lowerbound2)
                    else:
                        lowerBounds[i][centroid] = max(0, lowerBounds[i][centroid] - centroidDistanceChange[centroid])
                for i in self.pointsClassif[centroid]:
                    if prevDistanceChange[centroid] != 0:
                       
                        upperBounds[i] = (upperBounds[i] * oldCentroidDistanceChange[centroid] + oldUpperBounds[i] * centroidDistanceChange[centroid]) / prevDistanceChange[centroid]
                    else:
                        
                        upperBounds[i] += centroidDistanceChange[centroid]
                    upperBounds[i] = max(upperBounds[i], lowerBounds[i][centroid])

            oldLowerBounds = lowerBounds.copy()
            oldUpperBounds = upperBounds.copy()
            prevDistanceChange = centroidDistanceChange.copy()
            oldPrevCentroids = self.centroids.copy()

        for centroid in self.pointsClassif:
            for point in self.pointsClassif[centroid]:
                self.labels[point] = centroid