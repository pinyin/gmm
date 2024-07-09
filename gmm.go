package gmm

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"math"
)

// GaussianMixture represents a Gaussian Mixture Model
type GaussianMixture struct {
	NComponents int
	Weights     []float64
	Means       []*mat.VecDense
	Covariances []*mat.SymDense
	Precision   []*mat.SymDense
}

// NewGaussianMixture creates a new GaussianMixture instance
func NewGaussianMixture(nComponents int) *GaussianMixture {
	return &GaussianMixture{
		NComponents: nComponents,
		Weights:     make([]float64, nComponents),
		Means:       make([]*mat.VecDense, nComponents),
		Covariances: make([]*mat.SymDense, nComponents),
		Precision:   make([]*mat.SymDense, nComponents),
	}
}

// Fit trains the Gaussian Mixture Model on the given data
func (gmm *GaussianMixture) Fit(X *mat.Dense) error {
	// Step 1: Initialize parameters
	gmm.initializeParameters(X)

	// Step 2: Run EM algorithm
	maxIterations := 100
	tolerance := 1e-3
	prevLogLikelihood := math.Inf(-1)

	for i := 0; i < maxIterations; i++ {
		// E-step: Compute responsibilities
		responsibilities := gmm.eStep(X)

		// M-step: Update parameters
		gmm.mStep(X, responsibilities)

		// Compute log-likelihood
		logLikelihood := gmm.computeLogLikelihood(X)

		// Check for convergence
		if math.Abs(logLikelihood-prevLogLikelihood) < tolerance {
			break
		}
		prevLogLikelihood = logLikelihood
	}

	return nil
}

// initializeParameters initializes the GMM parameters
func (gmm *GaussianMixture) initializeParameters(X *mat.Dense) {
	_, nFeatures := X.Dims()

	// Initialize weights uniformly
	for i := range gmm.Weights {
		gmm.Weights[i] = 1.0 / float64(gmm.NComponents)
	}

	// Initialize means using k-means++ algorithm
	gmm.initializeMeans(X)

	// Initialize covariances as identity matrices
	for i := 0; i < gmm.NComponents; i++ {
		covData := make([]float64, nFeatures*nFeatures)
		precData := make([]float64, nFeatures*nFeatures)
		for j := 0; j < nFeatures; j++ {
			covData[j*nFeatures+j] = 1.0
			precData[j*nFeatures+j] = 1.0
		}
		gmm.Covariances[i] = mat.NewSymDense(nFeatures, covData)
		gmm.Precision[i] = mat.NewSymDense(nFeatures, precData)
	}
}

// initializeMeans initializes means using k-means++ algorithm
func (gmm *GaussianMixture) initializeMeans(X *mat.Dense) {
	nSamples, nFeatures := X.Dims()

	// Choose first centroid randomly
	firstIndex := rand.Intn(nSamples)
	firstCentroid := mat.NewVecDense(nFeatures, nil)
	firstCentroid.RowViewOf(X, firstIndex)
	gmm.Means[0] = firstCentroid

	// Choose remaining centroids
	for i := 1; i < gmm.NComponents; i++ {
		distances := make([]float64, nSamples)
		for j := 0; j < nSamples; j++ {
			minDist := math.Inf(1)
			for k := 0; k < i; k++ {
				dist := distanceSquared(X.RowView(j), gmm.Means[k])
				if dist < minDist {
					minDist = dist
				}
			}
			distances[j] = minDist
		}

		// Choose next centroid with probability proportional to distance
		sum := 0.0
		for _, d := range distances {
			sum += d
		}
		target := rand.Float64() * sum
		currentSum := 0.0
		for j, d := range distances {
			currentSum += d
			if currentSum >= target {
				newCentroid := mat.NewVecDense(nFeatures, nil)
				newCentroid.RowViewOf(X, j)
				gmm.Means[i] = newCentroid
				break
			}
		}
	}
}

// distanceSquared computes the squared Euclidean distance between two vectors
func distanceSquared(a, b mat.Vector) float64 {
	diff := mat.NewVecDense(a.Len(), nil)
	diff.SubVec(a, b)
	return mat.Dot(diff, diff)
}

// eStep computes the responsibilities (posterior probabilities)
func (gmm *GaussianMixture) eStep(X *mat.Dense) *mat.Dense {
	nSamples, _ := X.Dims()
	responsibilities := mat.NewDense(nSamples, gmm.NComponents, nil)

	for i := 0; i < nSamples; i++ {
		sample := X.RowView(i)
		totalProb := 0.0
		for j := 0; j < gmm.NComponents; j++ {
			prob := gmm.Weights[j] * gmm.componentDensity(sample, j)
			responsibilities.Set(i, j, prob)
			totalProb += prob
		}
		// Normalize responsibilities
		for j := 0; j < gmm.NComponents; j++ {
			responsibilities.Set(i, j, responsibilities.At(i, j)/totalProb)
		}
	}

	return responsibilities
}

// componentDensity computes the density of a sample for a given component
func (gmm *GaussianMixture) componentDensity(x mat.Vector, componentIndex int) float64 {
	nFeatures := x.Len()
	mean := gmm.Means[componentIndex]
	precision := gmm.Precision[componentIndex]

	diff := mat.NewVecDense(nFeatures, nil)
	diff.SubVec(x, mean)

	var mahalanobis float64
	mahalanobis = mat.Inner(diff, precision, diff)

	return math.Exp(-0.5*mahalanobis) / math.Sqrt(math.Pow(2*math.Pi, float64(nFeatures))*mat.Det(gmm.Covariances[componentIndex]))
}

// mStep updates the parameters based on the computed responsibilities
func (gmm *GaussianMixture) mStep(X *mat.Dense, responsibilities *mat.Dense) {
	nSamples, nFeatures := X.Dims()

	for j := 0; j < gmm.NComponents; j++ {
		nk := 0.0
		for i := 0; i < nSamples; i++ {
			nk += responsibilities.At(i, j)
		}

		// Update weights
		gmm.Weights[j] = nk / float64(nSamples)

		// Update means
		newMean := mat.NewVecDense(nFeatures, nil)
		for i := 0; i < nSamples; i++ {
			newMean.AddScaledVec(newMean, responsibilities.At(i, j), X.RowView(i))
		}
		newMean.ScaleVec(1/nk, newMean)
		gmm.Means[j] = newMean

		// Update covariances
		newCov := mat.NewSymDense(nFeatures, nil)
		for i := 0; i < nSamples; i++ {
			diff := mat.NewVecDense(nFeatures, nil)
			diff.SubVec(X.RowView(i), newMean)
			diffOuter := mat.NewSymDense(nFeatures, nil)
			diffOuter.SymOuterK(1, diff)
			for r := 0; r < nFeatures; r++ {
				for c := 0; c <= r; c++ {
					newVal := newCov.At(r, c) + responsibilities.At(i, j)*diffOuter.At(r, c)
					newCov.SetSym(r, c, newVal)
				}
			}
		}
		newCov.ScaleSym(1/nk, newCov)
		gmm.Covariances[j] = newCov

		// Update precision matrices
		var chol mat.Cholesky
		if ok := chol.Factorize(newCov); !ok {
			panic("covariance matrix is not positive definite")
		}
		gmm.Precision[j] = mat.NewSymDense(nFeatures, nil)
		err := chol.InverseTo(gmm.Precision[j])
		if err != nil {
			panic("failed to invert covariance matrix: " + err.Error())
		}
	}
}

// computeLogLikelihood computes the log-likelihood of the data
func (gmm *GaussianMixture) computeLogLikelihood(X *mat.Dense) float64 {
	nSamples, _ := X.Dims()
	logLikelihood := 0.0

	for i := 0; i < nSamples; i++ {
		sample := X.RowView(i)
		sampleLikelihood := 0.0
		for j := 0; j < gmm.NComponents; j++ {
			sampleLikelihood += gmm.Weights[j] * gmm.componentDensity(sample, j)
		}
		logLikelihood += math.Log(sampleLikelihood)
	}

	return logLikelihood
}

// Predict assigns samples to clusters
func (gmm *GaussianMixture) PredictProb(X *mat.Dense) [][]float64 {
	nSamples, _ := X.Dims()
	probabilities := make([][]float64, nSamples)

	responsibilities := gmm.eStep(X)
	for i := 0; i < nSamples; i++ {
		sampleProb := make([]float64, gmm.NComponents)
		for j := 0; j < gmm.NComponents; j++ {
			sampleProb[j] = responsibilities.At(i, j)
		}
		probabilities[i] = sampleProb
	}

	return probabilities
}

// Predict assigns samples to clusters
func (gmm *GaussianMixture) Predict(X *mat.Dense) []int {
	nSamples, _ := X.Dims()
	predictions := make([]int, nSamples)

	responsibilities := gmm.eStep(X)
	for i := 0; i < nSamples; i++ {
		maxProb := 0.0
		maxComponent := 0
		for j := 0; j < gmm.NComponents; j++ {
			prob := responsibilities.At(i, j)
			if prob > maxProb {
				maxProb = prob
				maxComponent = j
			}
		}
		predictions[i] = maxComponent
	}

	return predictions
}
