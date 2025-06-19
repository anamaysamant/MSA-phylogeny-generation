FAMILIES = ["PF00271","PF00005","PF00004","PF01535","PF13354","PF00595","PF00397","PF00153","PF07679",
            "PF00076","PF00072","PF00096","PF00512","PF00041","PF02518",
            "PF01356","PF03440","PF04008","PF06351","PF06355", "PF16747","PF18648"]
MSA_TYPES = ["seed"]
INIT_SEQS = ["0"]
PROPOSAL_TYPES = ["logits"]

num_simulations = 10

SIM_INDS = list(range(1,num_simulations+1))
SIM_INDS = list(map(str,SIM_INDS))

rule all:
    input:
        # expand("data/msa-{msa_type}-simulations/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal}-proposal/{fam}-{sim_ind}.fasta",
        #         msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS, init_seq = INIT_SEQS, proposal = PROPOSAL_TYPES),
        # expand("scores/msa-{msa_type}-simulations/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal}-proposal/{fam}-{sim_ind}.tsv",
        #         msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS, init_seq = INIT_SEQS, proposal = PROPOSAL_TYPES),
        # expand("data/msa-{msa_type}-simulation-trees/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal}-proposal/{fam}-{sim_ind}.newick",
        #         msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS, init_seq = INIT_SEQS, proposal = PROPOSAL_TYPES),
        expand("scores/msa-{msa_type}-sim-trees/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal}-proposal/{fam}-tree.tsv",
                msa_type = MSA_TYPES, fam = FAMILIES, init_seq = INIT_SEQS, proposal = PROPOSAL_TYPES),
        
# rule generate_tree:
#     input:
#         "data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta"
#     output:
#         "data/{msa_type}-trees/{fam}_{msa_type}.newick"
#     log:
#         "logs/generate-{msa_type}-trees/{fam}.log"
#     shell:
#         "FastTree {input} > {output}"

rule root_tree:
    input:
        "data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/{msa_type}-trees/{fam}_{msa_type}_rooted.newick"
    shell:
        "python scripts/root_tree.py --input {input} --output {output}"


rule generate_tree_ESM2:
    input:
        "data/msa-{msa_type}-simulations/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/{fam}-{sim_ind}.fasta"
    output:
        "data/msa-{msa_type}-simulation-trees/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/{fam}-{sim_ind}.newick"
    log:
        "logs/msa-{msa_type}-simulation-trees/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/{fam}-{sim_ind}.log"
    shell:
        "fasttree {input} > {output}"

rule generate_tree_metrics_ESM2:
    input:
        simulated_trees=expand("data/msa-{msa_type}-simulation-trees/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/{fam}-{sim_ind}.newick",
                sim_ind = SIM_INDS, allow_missing = True), 
        simulated_MSAs=expand("data/msa-{msa_type}-simulations/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/{fam}-{sim_ind}.fasta",
                sim_ind = SIM_INDS, allow_missing = True),
        seed_MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        seed_tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"     
    output:
        "scores/msa-{msa_type}-sim-trees/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/{fam}-tree.tsv"
    script:
        "scripts/tree_metrics_generator.py"

rule simulate_along_phylogeny_ESM2:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/msa-{msa_type}-simulations/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal}-proposal/{fam}-{sim_ind}.fasta"
    shell:
        """
        python scripts/simulate_along_phylogeny.py --output {output} --input_MSA {input.MSA} --input_tree {input.tree} \
        --tool ESM2 --seed {wildcards.sim_ind} --start_seq_index {wildcards.init_seq} --proposal_type {wildcards.proposal}
        """

rule generate_scores_ESM2:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/msa-{msa_type}-simulations/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal}-proposal/{fam}-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy",
        pdb_path="data/pdb-ref-structures/{fam}-ref.pdb"
    output:
        ungapped_seq=temp("data/msa-{msa_type}-simulations/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal}-proposal/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/msa-{msa_type}-simulations/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal}-proposal/{fam}-{sim_ind}.tbl"),
        score_table="scores/msa-{msa_type}-simulations/ESM2-33T-650M/{fam}/init-seq-{init_seq}/{proposal}-proposal/{fam}-{sim_ind}.tsv"
    shell:
       """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree} --pdb_path {input.pdb_path}
        """
