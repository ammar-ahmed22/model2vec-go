// io.go contains shared output helpers used by the CLI subcommands.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

// writeJSON serializes v as JSON. When path is empty the output is written to
// stdout; otherwise it is written to the given file path through a buffered
// writer. A trailing newline is always emitted (json.Encoder.Encode behavior),
// which makes the stdout form pipe-friendly.
func writeJSON(path string, v any) error {
	if path == "" {
		return json.NewEncoder(os.Stdout).Encode(v)
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer f.Close()

	bw := bufio.NewWriter(f)
	if err := json.NewEncoder(bw).Encode(v); err != nil {
		return fmt.Errorf("failed to write embeddings to JSON: %w", err)
	}
	if err := bw.Flush(); err != nil {
		return fmt.Errorf("failed to flush output file: %w", err)
	}
	return nil
}
