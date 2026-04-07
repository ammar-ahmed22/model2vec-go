// encode.go implements the `model2vec encode` subcommand, which embeds either
// a literal text string or every line of a file.
package main

import (
	"fmt"
	"os"
	"strings"

	model2vec "github.com/ammar-ahmed22/model2vec-go"
	"github.com/spf13/cobra"
)

// encodeOutput holds the value of the --output flag for the encode subcommand.
var encodeOutput string

// encodeCmd is the cobra command for `model2vec encode <input> <model>`.
//
// The first positional argument is interpreted as a path to a text file
// containing one sentence per line if such a file exists; otherwise it is
// treated as a single literal sentence. This matches the behavior of the
// model2vec-rs CLI's `Commands::Encode` branch.
var encodeCmd = &cobra.Command{
	Use:   "encode <input> <model>",
	Short: "Encode input texts into embeddings",
	Long: "Encode one or more sentences into embeddings using a Model2Vec model.\n\n" +
		"<input> is treated as a path to a UTF-8 text file (one sentence per line)\n" +
		"if such a file exists, and otherwise as a single literal sentence.\n" +
		"<model> is a HuggingFace Hub repository ID or a local model directory.\n\n" +
		"Examples:\n" +
		"  model2vec encode \"hello world\" minishlab/potion-base-8M\n" +
		"  model2vec encode sentences.txt ./local-model --output embs.json",
	Args: cobra.ExactArgs(2),
	RunE: runEncode,
}

func init() {
	encodeCmd.Flags().StringVarP(&encodeOutput, "output", "o", "",
		"Optional output file (JSON) for embeddings")
}

// runEncode is the RunE callback for encodeCmd.
func runEncode(cmd *cobra.Command, args []string) error {
	input, modelRef := args[0], args[1]

	texts, err := readInputTexts(input)
	if err != nil {
		return err
	}

	model, err := model2vec.FromPretrained(modelRef)
	if err != nil {
		return fmt.Errorf("failed to load model %q: %w", modelRef, err)
	}

	embeddings := model.Encode(texts)
	return writeJSON(encodeOutput, embeddings)
}

// readInputTexts returns the slice of sentences to encode given the raw
// <input> argument. If input refers to an existing regular file, the file is
// read and split on newlines (a single trailing empty line is dropped to match
// Rust's str::lines semantics). Otherwise input is returned as a single
// literal sentence.
func readInputTexts(input string) ([]string, error) {
	info, err := os.Stat(input)
	if err == nil && info.Mode().IsRegular() {
		raw, readErr := os.ReadFile(input)
		if readErr != nil {
			return nil, fmt.Errorf("failed to read input file %q: %w", input, readErr)
		}
		content := strings.TrimRight(string(raw), "\n")
		if content == "" {
			return []string{}, nil
		}
		return strings.Split(content, "\n"), nil
	}
	return []string{input}, nil
}
