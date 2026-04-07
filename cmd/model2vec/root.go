// Package main implements the model2vec command-line interface.
//
// The CLI mirrors the model2vec-rs binary 1:1 and exposes two subcommands:
//
//	model2vec encode        <input>    <model> [--output path]
//	model2vec encode-single <sentence> <model> [--output path]
//
// Both subcommands accept either a HuggingFace Hub repository ID
// (e.g. "minishlab/potion-base-8M") or a local directory containing a
// Model2Vec model as the <model> argument.
package main

import "github.com/spf13/cobra"

// rootCmd is the top-level cobra command for the model2vec binary.
var rootCmd = &cobra.Command{
	Use:   "model2vec",
	Short: "Model2Vec Go CLI",
	Long: "model2vec is a small command-line interface around the model2vec-go library.\n" +
		"It loads a Model2Vec static model from the HuggingFace Hub or a local path\n" +
		"and emits sentence embeddings as JSON.",
	SilenceUsage: true,
}

func init() {
	rootCmd.AddCommand(encodeCmd)
	rootCmd.AddCommand(encodeSingleCmd)
}
