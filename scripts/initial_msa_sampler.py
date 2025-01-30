
import argparse
from aux_msa_functions import *

parser = argparse.ArgumentParser()

parser.add_argument("-M", "--input_MSA", action="store", dest="input_MSA",
                    help="input protein family MSA")

parser.add_argument("-m", "--method", action="store", dest="method",
                    default="greedy", help="input protein family MSA")

parser.add_argument("-O", "--output", action="store", dest="output",
                    help="input protein family MSA")

parser.add_argument("-n", "--num_seqs", action="store", dest="num_seqs",
                    help="number of sequences to simulate", type=int
                )

args = parser.parse_args()

input_MSA = args.input_MSA
num_seqs = args.num_seqs
output = args.output
method = args.method


all_seqs = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(input_MSA, "fasta")]

if method == "greedy":
    sampled_MSA_tuples = greedy_select(all_seqs, num_seqs=num_seqs)

sampled_MSA_SeqRecords = [SeqRecord(Seq(record[1]), id = record[0], name= record[0], description= record[0]) for record in sampled_MSA_tuples]

with open(output, "w") as output_handle:
    SeqIO.write(sampled_MSA_SeqRecords, output_handle, "fasta")