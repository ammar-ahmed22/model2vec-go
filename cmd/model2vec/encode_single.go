// encode_single.go implements the `model2vec encode-single` subcommand, which
// embeds exactly one sentence supplied on the command line.
package main

import (
	"fmt"

	model2vec "github.com/ammar-ahmed22/model2vec-go"
	"github.com/spf13/cobra"
)

// encodeSingleOutput holds the value of the --output flag for the
// encode-single subcommand.
var encodeSingleOutput string

// encodeSingleCmd is the cobra command for
// `model2vec encode-single <sentence> <model>`.
var encodeSingleCmd = &cobra.Command{
	Use:   "encode-single <sentence> <model>",
	Short: "Encode a single sentence",
	Long: "Encode exactly one sentence into an embedding vector using a Model2Vec model.\n\n" +
		"<sentence> is always treated as literal text (never as a file path).\n" +
		"<model> is a HuggingFace Hub repository ID or a local model directory.\n\n" +
		"Examples:\n" +
		"  model2vec encode-single \"hello world\" minishlab/potion-base-8M\n" +
		"  model2vec encode-single \"hello world\" ./local-model --output emb.json",
	Args: cobra.ExactArgs(2),
	RunE: runEncodeSingle,
}

func init() {
	encodeSingleCmd.Flags().StringVarP(&encodeSingleOutput, "output", "o", "",
		"Optional output file (JSON) for the embedding")
}

// runEncodeSingle is the RunE callback for encodeSingleCmd.
func runEncodeSingle(cmd *cobra.Command, args []string) error {
	sentence, modelRef := args[0], args[1]

	model, err := model2vec.FromPretrained(modelRef)
	if err != nil {
		return fmt.Errorf("failed to load model %q: %w", modelRef, err)
	}

	embedding := model.EncodeSingle(sentence)
	return writeJSON(encodeSingleOutput, embedding)
}
