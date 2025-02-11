from Bio import SeqIO, Phylo
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-T", "--input", action="store", dest="input",
                    help="input protein family tree")

parser.add_argument("-O", "--output", action="store", dest="output",
                    help="rooted protein family tree"
                )

args = parser.parse_args()

tree_path = args.input
output = args.output

tree = Phylo.read(tree_path,"newick")
tree.root_at_midpoint()

Phylo.write(tree, output, "newick")

