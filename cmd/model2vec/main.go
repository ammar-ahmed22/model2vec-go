// Command model2vec is the command-line entry point for the model2vec-go
// library. See root.go for the cobra command tree and the individual
// encode*.go files for each subcommand's implementation.
package main

import "os"

func main() {
	if err := rootCmd.Execute(); err != nil {
		// cobra has already printed the error to stderr.
		os.Exit(1)
	}
}
