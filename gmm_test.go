package gmm

import (
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"math"
	"testing"
	"time"
)

func TestGaussianMixture(t *testing.T) {
	// Generate random trainData from a mixture of Gaussians
	nSamples := 1000
	nFeatures := 2
	nComponents := 3

	means := [][]float64{
		{0, 0},
		{5, 5},
		{-5, 5},
	}
	covs := []*mat.SymDense{
		mat.NewSymDense(nFeatures, []float64{1, 0, 0, 1}),
		mat.NewSymDense(nFeatures, []float64{1.5, 0.5, 0.5, 1.5}),
		mat.NewSymDense(nFeatures, []float64{2, -0.5, -0.5, 2}),
	}

	tolerance := 0.01

	// Create a random source
	src := rand.NewSource(uint64(time.Now().UnixNano()))

	trainData := mat.NewDense(nSamples, nFeatures, nil)
	testData := mat.NewDense(nSamples, nFeatures, nil)
	expectations := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		// Choose a component randomly based on weights
		chosenComponent := rand.Intn(nComponents)

		// Generate a Normal distribution for the chosen component
		dist, ok := distmv.NewNormal(means[chosenComponent], covs[chosenComponent], src)
		if !ok {
			t.Fatalf("Error creating normal distribution.")
		}

		trainSample := make([]float64, nFeatures)
		dist.Rand(trainSample)
		trainData.SetRow(i, trainSample)

		testSample := make([]float64, nFeatures)
		dist.Rand(testSample)
		testData.SetRow(i, testSample)

		expectations[i] = chosenComponent
	}

	// Fit GMM
	gmm := NewGaussianMixture(nComponents)
	err := gmm.Fit(trainData)
	if err != nil {
		t.Fatalf("Error fitting GMM: %v", err)
	}

	// Make predictions
	predictions := gmm.Predict(testData)
	t.Log(predictions)
	t.Log(expectations)

	// Test predictions against expectations
	if !predictionsMatchExpectations(expectations, predictions, tolerance) {
		t.Errorf("Predictions do not match expectations.")
	}

	// Check that the number of predicted clusters matches the number of components
	uniqueClusters := make(map[int]bool)
	for _, p := range predictions {
		uniqueClusters[p] = true
	}
	if len(uniqueClusters) != nComponents {
		t.Errorf("Expected %d clusters, but got %d", nComponents, len(uniqueClusters))
	}

	// Check that the weights sum to approximately 1
	weightSum := 0.0
	for _, w := range gmm.Weights {
		weightSum += w
	}
	if math.Abs(weightSum-1.0) > 1e-6 {
		t.Errorf("Weights do not sum to 1: %f", weightSum)
	}

	// Check that the covariance matrices are positive definite
	for i, cov := range gmm.Covariances {
		eigen := &mat.Eigen{}
		ok := eigen.Factorize(cov, mat.EigenRight)
		if !ok {
			t.Errorf("Covariance matrix %d is not positive definite", i)
		}
		values := eigen.Values(nil)
		for _, v := range values {
			if real(v) <= 0 {
				t.Errorf("Covariance matrix %d has non-positive eigenvalue", i)
			}
		}
	}
}

// Pair represents a pair of integers [a, b]
type Pair struct {
	A, B int
}

// ProbabilityMatrix represents the mapping from set A to set B
type ProbabilityMatrix map[int]map[int]float64

// CreateProbabilityMatrix creates a probability matrix from the given pairs
func CreateProbabilityMatrix(pairs []Pair) ProbabilityMatrix {
	matrix := make(ProbabilityMatrix)
	counts := make(map[int]map[int]int)

	// Count occurrences of each pair
	for _, pair := range pairs {
		if _, ok := counts[pair.A]; !ok {
			counts[pair.A] = make(map[int]int)
		}
		counts[pair.A][pair.B]++
	}

	// Calculate probabilities
	for a, bCounts := range counts {
		matrix[a] = make(map[int]float64)
		total := 0
		for _, count := range bCounts {
			total += count
		}
		for b, count := range bCounts {
			matrix[a][b] = float64(count) / float64(total)
		}
	}

	return matrix
}

// TestPredictions tests if the probability matrix meets the tolerance criterion
func predictionsMatchExpectations(a, b []int, tolerance float64) bool {
	if len(a) != len(b) {
		fmt.Println("Error: Slices 'a' and 'b' must have the same length")
		return false
	}

	pairs := make([]Pair, len(a))
	for i := range a {
		pairs[i] = Pair{A: a[i], B: b[i]}
	}

	matrix := CreateProbabilityMatrix(pairs)

	for _, bProbs := range matrix {
		maxProb := 0.0
		for _, prob := range bProbs {
			if prob > maxProb {
				maxProb = prob
			}
		}
		if maxProb < tolerance {
			return false
		}
	}

	return true
}
