import subprocess
from aux_msa_functions import *
from Bio import Phylo
from ete3 import Tree

import dendropy
from collections import Counter

def reorder_seqs(tree_root, all_nat_seqs_dict):

    output = []
    
    def reorder_seqs_recur(tree_root, all_nat_seqs_dict):
    
        b = tree_root.clades
        
        if len(b)>0:
            for clade in b:
               reorder_seqs_recur(clade, all_nat_seqs_dict) 
        else:
            counter = len(output)
            output.append((tree_root.name,all_nat_seqs_dict[tree_root.name]))

    reorder_seqs_recur(tree_root, all_nat_seqs_dict)

    return output

def rename_seqs(tree_root, all_nat_seqs_dict):
    
    def rename_seqs_recur(tree_root, all_nat_seqs_dict):
    
        b = tree_root.children
        
        if len(b)>0:
            for child in b:
               rename_seqs_recur(child, all_nat_seqs_dict) 
        else:
            tree_root.name = all_nat_seqs_dict[tree_root.name]

    rename_seqs_recur(tree_root, all_nat_seqs_dict)

def get_bipartitions(tree):

    bipartitions = {}
    leaves = set(tree.get_leaf_names())
    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            clade = frozenset(node.get_leaf_names())
            if 0 < len(clade) < len(leaves):  # Ignore full or empty set
                bipartitions[clade] = node.dist
    return bipartitions

def kuhner_felsenstein_distance(tree1, tree2):

    splits1 = get_bipartitions(tree1)
    splits2 = get_bipartitions(tree2)
    all_splits = set(splits1.keys()).union(splits2.keys())

    distance_squared = 0.0
    for split in all_splits:
        l1 = splits1.get(split, 0.0)
        l2 = splits2.get(split, 0.0)
        distance_squared += (l1 - l2) ** 2

    return distance_squared ** 0.5

def MAE_trees(tree1, tree2, leaf_names):

    distance_counts = 0
    error = 0
    for i in range(len(leaf_names)):
        for j in range(i+1, len(leaf_names)):
            tree1_dist = tree1.get_distance(leaf_names[i],leaf_names[j])
            tree2_dist = tree2.get_distance(leaf_names[i],leaf_names[j])

            error += abs(tree1_dist - tree2_dist)
            distance_counts += 1

    return error/distance_counts

def compute_clade_support(reference_tree_file, bootstrap_tree_files, rooted=False):
    taxon_namespace = dendropy.TaxonNamespace()

    # Load reference tree
    ref_tree = dendropy.Tree.get(
        path=reference_tree_file,
        schema="newick",
        rooting="force-unrooted" if not rooted else "default-rooted",
        taxon_namespace=taxon_namespace
    )
    ref_tree.encode_bipartitions()

    # Load bootstrap trees
    bootstrap_trees = []
    for f in bootstrap_tree_files:
        tree = dendropy.Tree.get(
            path=f,
            schema="newick",
            rooting="force-unrooted" if not rooted else "default-rooted",
            taxon_namespace=taxon_namespace
        )
        tree.encode_bipartitions()
        bootstrap_trees.append(tree)

    # Count bipartitions
    bipart_counts = Counter()
    for tree in bootstrap_trees:
        for edge in tree.postorder_edge_iter():
            bipart = edge.bipartition
            if bipart.is_trivial():
                continue
            bipart_counts[bipart.split_bitmask] += 1

    # Compute support
    support = {}
    total_bootstraps = len(bootstrap_trees)
    for edge in ref_tree.postorder_edge_iter():
        bipart = edge.bipartition
        if bipart.is_trivial():
            continue
        bitmask = bipart.split_bitmask
        count = bipart_counts.get(bitmask, 0)
        support_value = count / total_bootstraps

        # Resolve taxa from bitmask
        taxon_set = frozenset(
            str(taxon_namespace[i].label)
            for i in range(len(taxon_namespace))
            if bitmask & (1 << i)
        )
        support[taxon_set] = support_value

    support_values = list(support.values())
    median_support_value = np.median(support_values)
    return median_support_value


MSA_filename_seed = snakemake.input["seed_MSA"]
all_seqs_seed = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(MSA_filename_seed, "fasta")]
sim_tree_paths = list(snakemake.input["simulated_trees"])

sim_seqs_paths = list(snakemake.input["simulated_MSAs"])

seed_tree_path = snakemake.input["seed_tree"]
seed_tree = Tree(seed_tree_path,format=1)
R = seed_tree.get_midpoint_outgroup()
seed_tree.set_outgroup(R)

tree_biophylo = Phylo.read(seed_tree_path, "newick")
tree_biophylo.root_at_midpoint()

ordered_seqs = reorder_seqs(tree_biophylo.clade, dict(all_seqs_seed))
ordered_seqs_names = [seq[0] for seq in ordered_seqs]

metrics_df = []
sim_tree_paths_unrooted = []

for i in range(len(sim_tree_paths)):

    sim_ind = int(sim_tree_paths[i].split('-')[-1].split('.')[0])

    sim_tree = Tree(sim_tree_paths[i],format=1)
    R = sim_tree.get_midpoint_outgroup()
    sim_tree.set_outgroup(R)

    sim_seqs_names = [record.description for record in SeqIO.parse(sim_seqs_paths[i], "fasta")]
    mapping_dict = dict(zip(sim_seqs_names, ordered_seqs_names))
    rename_seqs(sim_tree, mapping_dict)

    sim_tree_unrooted = sim_tree.copy()
    sim_tree_unrooted.unroot()

    outfile = f"{sim_tree_paths[i]}.unrooted"
    sim_tree_unrooted.write(format=1, outfile=outfile)
    sim_tree_paths_unrooted.append(outfile)

    tree_metrics = seed_tree.compare(sim_tree)
    normalized_rf = tree_metrics["norm_rf"]
    kf_distance = kuhner_felsenstein_distance(seed_tree, sim_tree)
    mae = MAE_trees(seed_tree, sim_tree, leaf_names=list(mapping_dict.values()))

    metrics_df.append({"sim_ind":sim_ind, "normalized_rf":normalized_rf, "kf_distance":kf_distance, "MAE":mae})

metrics_df = pd.DataFrame(metrics_df)

metrics_df["median_clade_support"] = compute_clade_support(seed_tree_path, sim_tree_paths_unrooted, rooted = False)

for path in sim_tree_paths_unrooted:
    subprocess.run(["rm", path])

metrics_df.to_csv(snakemake.output[0], sep="\t", index=False)




