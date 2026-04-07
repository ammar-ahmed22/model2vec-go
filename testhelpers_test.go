package model2vec_test

import (
	"math"
	"testing"

	model2vec "github.com/ammar-ahmed22/model2vec-go"
)

// assertRelativeEq fails the test if got and want differ by more than maxRel
// in relative terms. Mirrors approx::assert_relative_eq! semantics: when both
// values are zero (or near it) the assertion passes; otherwise the absolute
// difference must be ≤ maxRel * max(|got|, |want|).
func assertRelativeEq(t *testing.T, got, want float32, maxRel float64, ctx string) {
	t.Helper()
	diff := math.Abs(float64(got - want))
	scale := math.Max(math.Abs(float64(got)), math.Abs(float64(want)))
	if scale == 0 {
		return
	}
	if diff/scale > maxRel {
		t.Errorf("%s: got %v want %v (rel %v > %v)", ctx, got, want, diff/scale, maxRel)
	}
}

// loadFixture loads a fixture model from testdata/<name>.
func loadFixture(t *testing.T, name string) *model2vec.StaticModel {
	t.Helper()
	path := "testdata/" + name
	m, err := model2vec.FromPretrained(path)
	if err != nil {
		t.Fatalf("FromPretrained(%q): %v", path, err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}
