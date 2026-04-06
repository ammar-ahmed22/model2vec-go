// model2vec is the command-line interface for model2vec-go.
// CLI support is planned for a future release.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "model2vec CLI is not yet implemented.")
	fmt.Fprintln(os.Stderr, "Use the library directly:")
	fmt.Fprintln(os.Stderr, `  import model2vec "github.com/ammar-ahmed22/model2vec-go"`)
	os.Exit(1)
}
